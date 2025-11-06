from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging, uuid, re

from app.core.database import get_db, SessionLocal
from app.core.auth import get_current_user_dev
from app.core.config import settings

from app.models.training import TrainingJob, TrainingStatus
from app.models.model import Model, ModelStatus, ModelFramework
from app.utils.project_helper import get_or_create_default_project

from app.schemas.training import (
    TrainingJobCreate, TrainingJobUpdate, TrainingJobResponse
)
from app.rabbitmq.producer import send_train_request

logger = logging.getLogger(__name__)
router = APIRouter()

def _norm_prefix(p: str) -> str:
    return p if p.endswith("/") else p + "/"

def _slugify_model_name(s: str) -> str:
    # "YOLOv8n" -> "yolov8n", 공백/특수문자 제거(언더스코어로 대체)
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]", "", s)
    return s or "model"

@router.post("/", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    # 이름 중복 방지
    if db.query(TrainingJob).filter(TrainingJob.name == job.name).first():
        raise HTTPException(status_code=400, detail="Training job with this name already exists")

    # DB Job 생성
    hp = {**job.hyperparameters, "dataset_name": job.dataset_name, "dataset_s3_prefix": job.dataset_s3_prefix}
    total_epochs = hp.get("epochs", 20)

    db_job = TrainingJob(
        name=job.name,
        dataset_id=None,  # dataset_id 미사용
        architecture=job.architecture or hp.get("model", "yolov8n"),  # 조회용/백업용
        hyperparameters=hp,
        total_epochs=total_epochs,
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job); db.commit(); db.refresh(db_job)

    # Model 생성/연결
    model_name = f"{job.name}_model"
    model = db.query(Model).filter(Model.name == model_name).first()
    if model:
        model.status = ModelStatus.TRAINING
        model.training_config = hp
        model.hyperparameters = hp
        db.commit(); db.refresh(model)
    else:
        default_project = get_or_create_default_project(db, current_user["uid"])
        model = Model(
            name=model_name,
            architecture=db_job.architecture,
            project_id=default_project.id,
            framework=ModelFramework.PYTORCH,
            status=ModelStatus.TRAINING,
            training_config=hp,
            hyperparameters=hp,
            created_by=current_user["uid"]
        )
        db.add(model); db.commit(); db.refresh(model)

    db_job.model_id = model.id
    db.commit(); db.refresh(db_job)

    # 외부용 job_id(문자열) 생성 & 저장
    job_id_str = uuid.uuid4().hex[:8]
    db_job.hyperparameters["external_job_id"] = job_id_str
    db.commit(); db.refresh(db_job)

    # payload 구성
    try:
        s3_prefix = _norm_prefix(job.dataset_s3_prefix)
        dataset_name = job.dataset_name
        model_key = _slugify_model_name(hp.get("model") or db_job.architecture)

        payload = {
            "job_id": job_id_str,
            "dataset": {
                "s3_prefix": s3_prefix,
                "name": dataset_name
            },
            "output": {
                "prefix": f"models/{model_key}",
                "model_name": f"{dataset_name}.pt"
            },
            "hyperparams": {
                "model": model_key,
                "epochs": hp.get("epochs", 20),
                "batch":  hp.get("batch", 8),
                "imgsz":  hp.get("imgsz", hp.get("img_size", 640))
            }
        }

        send_train_request(payload)

        db_job.status = TrainingStatus.RUNNING
        db.commit(); db.refresh(db_job)
        return db_job

    except Exception as e:
        db_job.status = TrainingStatus.FAILED
        db_job.training_logs = f"Failed to start training: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))