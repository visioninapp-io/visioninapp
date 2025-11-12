from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
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

# LLM 기반 학습 파이프라인 (llm 모듈에서 import)
from llm.graph.training.builder import builder

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
    job_id_str = uuid.uuid4()
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


@router.post("/llm", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_llm_training(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    LLM 기반 AI Modal Training
    사용자의 자연어 요청을 분석하여 최적의 학습 파라미터 자동 설정
    """
    # 프론트엔드에서 user_query 또는 query로 보낼 수 있음
    user_query = request.get("user_query") or request.get("query", "")
    user_query = user_query.strip() if user_query else ""
    dataset_name = request.get("dataset_name", "")
    dataset_s3_prefix = request.get("dataset_s3_prefix", "")
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Dataset name is required")
    
    # TrainingJob 생성
    job_name = f"AI-{dataset_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    db_job = TrainingJob(
        name=job_name,
        dataset_id=None,
        architecture="AI-Auto",  # LLM이 자동으로 결정
        hyperparameters={
            "user_query": user_query,
            "dataset_name": dataset_name,
            "dataset_s3_prefix": dataset_s3_prefix,
            "ai_mode": True
        },
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # LangGraph 학습 파이프라인을 백그라운드에서 실행
    background_tasks.add_task(
        run_llm_training_pipeline,
        job_id=db_job.id,
        user_query=user_query,
        dataset_path=dataset_name  # or dataset_s3_prefix
    )
    
    # 상태를 RUNNING으로 변경
    db_job.status = TrainingStatus.RUNNING
    db.commit()
    db.refresh(db_job)
    
    # TrainingJobResponse 생성 (hyperparameters에서 dataset 정보 추출)
    hp = db_job.hyperparameters or {}
    response_data = TrainingJobResponse(
        id=db_job.id,
        name=db_job.name,
        architecture=db_job.architecture,
        model_id=db_job.model_id,
        dataset_name=hp.get("dataset_name", dataset_name),
        dataset_s3_prefix=hp.get("dataset_s3_prefix", dataset_s3_prefix),
        hyperparameters=hp,
        status=db_job.status,
        s3_log_uri=getattr(db_job, "s3_log_uri", None),
        external_job_id=hp.get("external_job_id"),
        created_at=db_job.created_at,
        created_by=db_job.created_by,
        completed_at=db_job.completed_at,
        error_message=db_job.error_message,
    )
    
    return response_data


def run_llm_training_pipeline(job_id: int, user_query: str, dataset_path: str):
    """
    백그라운드에서 LangGraph 기반 학습 실행
    """
    import asyncio
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    job = None
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logger.error(f"[LLM Training] Job {job_id} not found")
            return
            
        logger.info(f"[LLM Training] Starting job {job_id}: {user_query}")
        
        # LLM 기반 학습 파이프라인 실행
        # builder 함수에 job_id도 전달 (builder 함수 시그니처: builder(user_query, dataset_path, job_id))
        job_id_str = str(job_id)
        builder(user_query, dataset_path, job_id_str)
        
        job.status = TrainingStatus.COMPLETED
        job.training_logs = "AI training completed successfully"
        db.commit()
        logger.info(f"[LLM Training] Job {job_id} completed")
        
    except (asyncio.CancelledError, KeyboardInterrupt):
        # 서버 종료 시 백그라운드 태스크가 취소되는 경우 조용히 종료
        logger.info(f"[LLM Training] Job {job_id} cancelled (server shutdown)")
        if job:
            job.status = TrainingStatus.FAILED
            job.training_logs = "Training cancelled due to server shutdown"
            try:
                db.commit()
            except Exception:
                pass  # DB 연결이 이미 끊어진 경우 무시
        return  # 조용히 종료
        
    except Exception as e:
        logger.error(f"[LLM Training] Job {job_id} failed: {e}", exc_info=True)
        if job:
            try:
                job.status = TrainingStatus.FAILED
                job.training_logs = f"AI Training failed: {str(e)}"
                db.commit()
            except Exception:
                pass  # DB 연결이 이미 끊어진 경우 무시
    finally:
        try:
            db.close()
        except Exception:
            pass  # DB 연결이 이미 끊어진 경우 무시