from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging, uuid, re

from app.core.database import get_db, SessionLocal
from app.core.auth import get_current_user
from app.core.config import settings

from app.models.training import TrainingJob, TrainingStatus
from app.models.model import Model  # ✅ ERD의 model 테이블 (status/framework 없음)
from app.models.model_version import ModelVersion  # ✅ ERD의 model_version 테이블
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


def _get_or_create_model(db: Session, *, name: str, project_id: int, task: str | None = None, description: str | None = None) -> Model:
    """
    ERD의 model 테이블만 사용 (status/framework 없음).
    동일 name이면 재사용. 없으면 생성.
    """
    model = db.query(Model).filter(Model.name == name, Model.project_id == project_id).first()
    if model:
        return model
    model = Model(
        project_id=project_id,
        name=name,
        task=task,
        description=description,
        created_at=datetime.utcnow()
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _get_or_create_model_v0(db: Session, *, model_id: int, framework: str | None, framework_version: str | None, training_config: dict) -> ModelVersion:
    """
    v0 고정 정책:
      - model_version은 해당 model에 대해 항상 1개만 사용 (v0)
      - 있으면 재사용(UPDATE), 없으면 최초 생성(INSERT)
    ERD 필드:
      id, model_id, dataset_version_id, parent_model_version_id, init_from_artifact_id,
      framework(VARCHAR50), framework_version(VARCHAR50), training_config(JSON),
      incremental_info(JSON), is_frozen(BOOLEAN), created_at(TIMESTAMP)
    """
    mv = db.query(ModelVersion).filter(ModelVersion.model_id == model_id).first()
    if mv:
        # 재사용(v0) → 최신 학습 설정만 반영 (필요시 다른 필드도 업데이트)
        mv.framework = (framework or mv.framework)
        mv.framework_version = (framework_version or mv.framework_version)
        mv.training_config = training_config
        # mv.dataset_version_id = mv.dataset_version_id  # 필요 시 유지/변경
        db.commit()
        db.refresh(mv)
        return mv

    # 최초 생성(v0)
    mv = ModelVersion(
        model_id=model_id,
        dataset_version_id=None,            # ✅ 데이터셋 버전 v0를 별도로 쓰면 이곳에 v0 id를 넣으세요
        parent_model_version_id=None,
        init_from_artifact_id=None,
        framework=(framework or "pytorch"),
        framework_version=(framework_version or "unknown"),
        training_config=training_config,
        incremental_info=None,
        is_frozen=False,
        created_at=datetime.utcnow()
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
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

    # 1) 하이퍼파라미터 합치기(프론트 값 우선, 없을 때만 기본값 적용)
    hp = {
        **(job.hyperparameters or {}),
        "dataset_name": job.dataset_name,
        "dataset_s3_prefix": job.dataset_s3_prefix
    }
    total_epochs = hp.get("epochs", 20)

    # 2) TrainingJob 생성
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

    # 3) Model(v0 고정)에 연결
    #    - model 테이블: ERD 그대로 사용(필요 최소 필드만)
    default_project = get_or_create_default_project(db, current_user["uid"])
    model_name = f"{job.name}_model"
    model = _get_or_create_model(
        db,
        name=model_name,
        project_id=default_project.id,
        task=None,               # 필요시 'object_detection' 등 지정
        description=None
    )

    # 4) ModelVersion: v0만 사용 (있으면 UPDATE, 없으면 1회 INSERT)
    #    - framework / framework_version 은 문자열(VARCHAR)만 허용
    framework_str = (job.architecture or hp.get("model", "pytorch"))
    framework_version_str = getattr(settings, "FRAMEWORK_VERSION", None) or "unknown"

    mv = _get_or_create_model_v0(
        db,
        model_id=model.id,
        framework=framework_str,
        framework_version=framework_version_str,
        training_config=hp
    )

    # 5) TrainingJob - 모델 참조만 저장(스키마 범위 내)
    db_job.model_id = model.id
    # (TrainingJob에 model_version_id 컬럼이 있다면 여기에 mv.id를 추가로 저장하세요)
    db.commit(); db.refresh(db_job)

    # 6) 외부용 job_id 생성 & TrainingJob.hyperparameters에 보관
    job_id_str = uuid.uuid4().hex[:8]
    db_job.hyperparameters["external_job_id"] = job_id_str
    db.commit(); db.refresh(db_job)

    # 7) MQ payload 구성 (프론트 값 우선, 없을 때만 기본값)
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
                "epochs": hp.get("epochs"),
                "batch":  hp.get("batch"),
                "imgsz":  hp.get("imgsz", hp.get("img_size"))
            }
            # 필요하다면 model_version 정보도 포함해 GPU에서 아티팩트에 쓰게 할 수 있음:
            # ,"model_info": {"model_id": model.id, "model_version_id": mv.id}
        }

        # 8) 학습 트리거 발행 (jobs.cmd:train.start)
        send_train_request(payload)

        # 9) 상태 전이
        db_job.status = TrainingStatus.RUNNING
        db.commit(); db.refresh(db_job)
        return db_job

    except Exception as e:
        db_job.status = TrainingStatus.FAILED
        db_job.training_logs = f"Failed to start training: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))