from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
import logging, uuid, re

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.config import settings

from app.models.training import TrainingJob, TrainingStatus
from app.models.model import Model
from app.models.model_version import ModelVersion
from app.models.model_artifact import ModelArtifact
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
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]", "", s)
    return s or "model"


def _get_or_create_model(db: Session, *, name: str, project_id: int, task: str | None = None, description: str | None = None) -> Model:
    m = db.query(Model).filter(Model.name == name, Model.project_id == project_id).first()
    if m:
        return m
    m = Model(project_id=project_id, name=name, task=task, description=description, created_at=datetime.utcnow())
    db.add(m); db.commit(); db.refresh(m)
    return m


def _get_or_create_model_v0(db: Session, *, model_id: int, framework: str | None, framework_version: str | None, training_config: dict) -> ModelVersion:
    mv = db.query(ModelVersion).filter(ModelVersion.model_id == model_id).first()
    if mv:
        mv.framework = (framework or mv.framework)
        mv.framework_version = (framework_version or mv.framework_version)
        mv.training_config = training_config
        db.commit(); db.refresh(mv)
        return mv

    mv = ModelVersion(
        model_id=model_id,
        dataset_version_id=None,
        parent_model_version_id=None,
        init_from_artifact_id=None,
        framework=(framework or "pytorch"),
        framework_version=(framework_version or "unknown"),
        training_config=training_config,
        incremental_info=None,
        is_frozen=False,
        created_at=datetime.utcnow()
    )
    db.add(mv); db.commit(); db.refresh(mv)
    return mv


@router.post("/", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # 0) 이름 중복 방지
    if db.query(TrainingJob).filter(TrainingJob.name == job.name).first():
        raise HTTPException(status_code=400, detail="Training job with this name already exists")

    # 1) HP 병합
    hp = {
        **(job.hyperparameters or {}),
        "dataset_name": job.dataset_name,
        "dataset_s3_prefix": job.dataset_s3_prefix
    }
    total_epochs = hp.get("epochs", 20)

    # 2) job 생성
    db_job = TrainingJob(
        name=job.name,
        dataset_id=None,
        architecture=job.architecture or hp.get("model", "yolov8n"),
        hyperparameters=hp,
        total_epochs=total_epochs,
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job); db.commit(); db.refresh(db_job)

    # 3) model / v0
    default_project = get_or_create_default_project(db, current_user["uid"])
    model = _get_or_create_model(
        db, name=f"{job.name}_model", project_id=default_project.id, task=None, description=None
    )
    framework_str = (job.architecture or hp.get("model", "pytorch"))
    framework_version_str = getattr(settings, "FRAMEWORK_VERSION", None) or "unknown"
    mv = _get_or_create_model_v0(
        db, model_id=model.id, framework=framework_str, framework_version=framework_version_str, training_config=hp
    )

    # 4) job ↔ model 연결
    db_job.model_id = model.id
    db.commit(); db.refresh(db_job)

    # 5) external_job_id
    job_id_str = uuid.uuid4().hex[:8]
    db_job.hyperparameters["external_job_id"] = job_id_str
    db.commit(); db.refresh(db_job)

    # 6) 아티팩트(PT) 선점 INSERT (storage_uri만 확정)
    try:
        s3_prefix = _norm_prefix(job.dataset_s3_prefix)
        dataset_name = job.dataset_name
        model_key = _slugify_model_name(hp.get("model") or db_job.architecture)

        object_key = f"models/{model_key}/{dataset_name}.pt"  # 버킷 제외 S3 key
        # 중복 방지: 같은 v0에서 같은 key가 있으면 재사용
        artifact = db.query(ModelArtifact).filter(
            ModelArtifact.model_version_id == mv.id,
            ModelArtifact.storage_uri == object_key
        ).first()
        if not artifact:
            artifact = ModelArtifact(
                model_version_id=mv.id,
                storage_uri=object_key,
                format="pt"  # 참고용, 현재는 NULL이어도 되지만 남겨둠
            )
            db.add(artifact); db.commit(); db.refresh(artifact)

        # 7) MQ payload 구성 & 발행
        payload = {
            "job_id": job_id_str,
            "dataset": {"s3_prefix": s3_prefix, "name": dataset_name},
            "output":  {"prefix": f"models/{model_key}", "model_name": f"{dataset_name}.pt"},
            "hyperparams": {
                "model": model_key,
                "epochs": hp.get("epochs"),
                "batch":  hp.get("batch"),
                "imgsz":  hp.get("imgsz", hp.get("img_size"))
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
