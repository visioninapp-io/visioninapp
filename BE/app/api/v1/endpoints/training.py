from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import logging, uuid, re
import asyncio
import boto3
from botocore.exceptions import ClientError

from app.core.database import get_db, SessionLocal
from app.core.auth import get_current_user
from app.core.config import settings

from app.models.training import TrainingJob, TrainingStatus
from app.models.model import Model
from app.models.model_version import ModelVersion
from app.models.model_artifact import ModelArtifact
from app.models.dataset import Dataset
from app.utils.project_helper import get_or_create_default_project
from app.utils.dataset_helper import get_or_create_dataset_version

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


def _get_next_version(s3_client, bucket: str, dataset_name: str, train_type: str) -> str:
    """
    S3에서 해당 dataset의 train_type 폴더 내 최신 버전을 조회하여 다음 버전 반환
    train_type: "train" 또는 "ai-train"
    반환: "v1", "v2", "v3"...
    """
    try:
        prefix = f"models/{dataset_name}/{train_type}/"
        logger.info(f"[Version] Checking S3 for existing versions: {prefix}")
        
        # S3에서 해당 prefix의 폴더 목록 조회
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter='/'
        )
        
        # CommonPrefixes에서 버전 폴더들 추출
        versions = []
        if 'CommonPrefixes' in response:
            for obj in response['CommonPrefixes']:
                folder_name = obj['Prefix'].rstrip('/').split('/')[-1]  # 마지막 폴더명 (예: v1, v2)
                if folder_name.startswith('v') and folder_name[1:].isdigit():
                    version_num = int(folder_name[1:])
                    versions.append(version_num)
        
        # 다음 버전 계산
        if versions:
            next_version = max(versions) + 1
            logger.info(f"[Version] Found existing versions: {sorted(versions)}, next: v{next_version}")
        else:
            next_version = 1
            logger.info(f"[Version] No existing versions found, starting with v1")
        
        return f"v{next_version}"
        
    except ClientError as e:
        logger.warning(f"[Version] Error checking S3 versions: {e}, defaulting to v1")
        return "v1"
    except Exception as e:
        logger.warning(f"[Version] Unexpected error checking versions: {e}, defaulting to v1")
        return "v1"


def _get_or_create_model(db: Session, *, name: str, project_id: int, task: str | None = None, description: str | None = None) -> Model:
    m = db.query(Model).filter(Model.name == name, Model.project_id == project_id).first()
    if m:
        return m
    m = Model(project_id=project_id, name=name, task=task, description=description, created_at=datetime.utcnow())
    db.add(m); db.commit(); db.refresh(m)
    return m


def _get_or_create_model_v0(db: Session, *, model_id: int, dataset_version_id: int, framework: str | None, framework_version: str | None, training_config: dict) -> ModelVersion:
    mv = db.query(ModelVersion).filter(ModelVersion.model_id == model_id).first()
    if mv:
        mv.framework = (framework or mv.framework)
        mv.framework_version = (framework_version or mv.framework_version)
        mv.training_config = training_config
        db.commit(); db.refresh(mv)
        return mv

    mv = ModelVersion(
        model_id=model_id,
        dataset_version_id=dataset_version_id,
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


@router.get("/", response_model=List[TrainingJobResponse])
async def get_training_jobs(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all training jobs for current user"""
    jobs = db.query(TrainingJob).filter(
        TrainingJob.created_by == current_user["uid"]
    ).order_by(TrainingJob.created_at.desc()).all()
    return jobs


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a training job and optionally its associated model (from both DB and S3)"""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.created_by == current_user["uid"]
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Initialize S3 client for potential S3 cleanup
    s3_client = None
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
    except Exception as e:
        logger.warning(f"[Training] Failed to initialize S3 client: {e}")
    
    # Delete associated model if exists (including S3 artifacts)
    if job.model_id:
        model = db.query(Model).filter(Model.id == job.model_id).first()
        if model:
            try:
                # Get all model versions and their artifacts
                model_versions = db.query(ModelVersion).filter(
                    ModelVersion.model_id == job.model_id
                ).all()
                
                # Delete all artifacts from S3
                if s3_client:
                    for version in model_versions:
                        for artifact in version.artifacts:
                            if artifact.storage_uri:
                                try:
                                    logger.info(f"[Training] Deleting S3 object: {artifact.storage_uri}")
                                    s3_client.delete_object(
                                        Bucket=settings.AWS_BUCKET_NAME,
                                        Key=artifact.storage_uri
                                    )
                                    logger.info(f"[Training] Successfully deleted from S3: {artifact.storage_uri}")
                                except ClientError as e:
                                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                                    if error_code != 'NoSuchKey':
                                        logger.warning(f"[Training] Failed to delete S3 object {artifact.storage_uri}: {e}")
                                    # Continue even if S3 deletion fails (file might not exist)
                                except Exception as e:
                                    logger.warning(f"[Training] Error deleting S3 object {artifact.storage_uri}: {e}")
                                    # Continue even if S3 deletion fails
                    
                    # Also delete results.csv if exists (using new path structure)
                    try:
                        # 새 경로 구조: models/{dataset_name}/{train_type}/{version}/results.csv
                        hp = job.hyperparameters or {}
                        dataset_name = hp.get('dataset_name')
                        version = hp.get('version')
                        is_ai = hp.get('ai_mode', False)
                        
                        if dataset_name and version:
                            train_type = 'ai-train' if is_ai else 'train'
                            results_csv_key = f"models/{dataset_name}/{train_type}/{version}/results.csv"
                            
                            s3_client.delete_object(
                                Bucket=settings.AWS_BUCKET_NAME,
                                Key=results_csv_key
                            )
                            logger.info(f"[Training] Successfully deleted results.csv from S3: {results_csv_key}")
                        else:
                            logger.warning(f"[Training] Cannot determine S3 path for job {job.id} - missing dataset_name or version")
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                        if error_code != 'NoSuchKey':
                            logger.warning(f"[Training] Failed to delete results.csv from S3: {e}")
                    except Exception as e:
                        logger.warning(f"[Training] Error deleting results.csv from S3: {e}")
                
                # Delete model from database (cascade will handle versions and artifacts)
                db.delete(model)
                # 모델 삭제를 먼저 commit하여 확실히 삭제
                db.commit()
                logger.info(f"[Training] Deleted associated model {job.model_id} for job {job_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"[Training] Error deleting model {job.model_id}: {e}", exc_info=True)
                # 모델 삭제 실패 시 job 삭제도 중단
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete associated model: {str(e)}"
                )
    
    # Even if model_id is None, try to delete potential S3 files based on hyperparameters
    # This handles cases like LLM training where model might not be created yet
    if not job.model_id and s3_client:
        try:
            # 새 경로 구조로 삭제 시도
            hp = job.hyperparameters or {}
            dataset_name = hp.get('dataset_name')
            version = hp.get('version')
            is_ai = hp.get('ai_mode', False)
            
            if dataset_name and version:
                train_type = 'ai-train' if is_ai else 'train'
                results_csv_key = f"models/{dataset_name}/{train_type}/{version}/results.csv"
                
                s3_client.delete_object(
                    Bucket=settings.AWS_BUCKET_NAME,
                    Key=results_csv_key
                )
                logger.info(f"[Training] Successfully deleted results.csv from S3 (no model_id): {results_csv_key}")
            else:
                logger.warning(f"[Training] Cannot determine S3 path for job {job.id} (no model_id) - missing hyperparameters")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code != 'NoSuchKey':
                logger.warning(f"[Training] Failed to delete results.csv from S3 (no model_id): {e}")
            # File might not exist, which is fine
        except Exception as e:
            logger.warning(f"[Training] Error deleting results.csv from S3 (no model_id): {e}")
    
    # Delete the training job from database
    db.delete(job)
    db.commit()
    
    logger.info(f"[Training] Deleted training job {job_id} (name: {job.name}) from DB and S3")
    return None


@router.post("/", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # 0) 이름 중복 방지
    if db.query(TrainingJob).filter(TrainingJob.name == job.name).first():
        raise HTTPException(status_code=400, detail="Training job with this name already exists")

    # 1) Dataset 조회
    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with id {job.dataset_id} not found")

    dataset_name = dataset.name
    dataset_s3_prefix = f"datasets/{dataset_name}/"

    # S3 client 초기화 (버전 확인용)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    
    # 다음 버전 번호 가져오기
    version = _get_next_version(s3_client, settings.AWS_BUCKET_NAME, dataset_name, "train")

    # 2) HP 병합
    hp = {
        **(job.hyperparameters or {}),
        "dataset_name": dataset_name,
        "dataset_s3_prefix": dataset_s3_prefix,
        "version": version,
        "output_prefix": f"models/{dataset_name}/train/{version}"
    }
    total_epochs = hp.get("epochs", 20)

    # 3) job 생성
    db_job = TrainingJob(
        name=job.name,
        dataset_id=job.dataset_id,
        architecture=job.architecture,
        hyperparameters=hp,
        total_epochs=total_epochs,
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job); db.commit(); db.refresh(db_job)

    # 4) dataset_version 생성/조회 (v0로 고정)
    dataset_version = get_or_create_dataset_version(db, job.dataset_id, version_tag="v0")

    # 5) model / v0
    default_project = get_or_create_default_project(db, current_user["uid"])
    model = _get_or_create_model(
        db, name=job.name, project_id=default_project.id, task=None, description=f"Model for {job.name}"
    )
    framework_str = job.architecture
    framework_version_str = getattr(settings, "FRAMEWORK_VERSION", None) or "unknown"
    mv = _get_or_create_model_v0(
        db, model_id=model.id, dataset_version_id=dataset_version.id, framework=framework_str, framework_version=framework_version_str, training_config=hp
    )

    # 5) job ↔ model 연결
    db_job.model_id = model.id
    db.commit(); db.refresh(db_job)

    # 6) external_job_id
    job_id_str = str(uuid.uuid4()).replace("-", "")
    db_job.hyperparameters["external_job_id"] = job_id_str
    db.commit(); db.refresh(db_job)

    # 7) 아티팩트(PT) 선점 INSERT (storage_uri만 확정)
    try:
        # 새로운 경로 구조: models/{dataset_name}/train/{version}/best.pt
        object_key = f"models/{dataset_name}/train/{version}/best.pt"
        
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

        # 8) MQ payload 구성 & 발행 (GPU가 기대하는 형식)
        # Check if using a pretrained model
        pretrained_model_path = None
        pretrained_model_id = hp.get("pretrained_model_id")
        if pretrained_model_id:
            # Load pretrained model artifact path from model ID
            pretrained_model = db.query(Model).filter(Model.id == pretrained_model_id).first()
            if pretrained_model:
                # Find the latest artifact for this model
                # Explicitly specify join condition: ModelArtifact.model_version_id == ModelVersion.id
                pretrained_artifact = db.query(ModelArtifact).join(
                    ModelVersion, ModelArtifact.model_version_id == ModelVersion.id
                ).filter(
                    ModelVersion.model_id == pretrained_model.id
                ).order_by(ModelArtifact.created_at.desc()).first()
                if pretrained_artifact:
                    pretrained_model_path = pretrained_artifact.storage_uri
                    logger.info(f"[Training] Using pretrained model: {pretrained_model_path}")

        payload = {
            "job_id": job_id_str,
            "dataset": {
                "s3_prefix": dataset_s3_prefix,
                "name": dataset_name
            },
            "output": {
                "prefix": f"models/{dataset_name}/train/{version}",  # 새로운 경로 구조
                "model_name": "best.pt",
                "metrics_name": "results.csv"
            },
            "hyperparams": {
                "model": job.architecture,
                "epochs": hp.get("epochs", 20),
                "batch": hp.get("batch_size", hp.get("batch", 8)),
                "imgsz": hp.get("img_size", hp.get("imgsz", 640)),
                "pretrained": pretrained_model_path  # Pass pretrained model S3 path if specified
            },
            "split": [0.8, 0.1, 0.1],
            "split_seed": 42,
            "move_files": False,
            "user_id": current_user["uid"],  # Add user_id to payload
            "model_id": db_job.model_id      # Add model_id to payload
        }
        send_train_request(payload)

        db_job.status = TrainingStatus.RUNNING
        db.commit(); db.refresh(db_job)
        return db_job

    except Exception as e:
        db_job.status = TrainingStatus.FAILED
        db_job.training_log = f"Failed to start training: {e}"
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
    job_name = request.get("job_name", "").strip()
    dataset_id = request.get("dataset_id")
    dataset_name = request.get("dataset_name", "")
    dataset_s3_prefix = request.get("dataset_s3_prefix", "")
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    if not job_name:
        raise HTTPException(status_code=400, detail="Job name is required")
    
    # Dataset 조회 (dataset_id 우선, 없으면 dataset_name으로 찾기)
    dataset = None
    if dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found")
        dataset_name = dataset.name
    elif dataset_name:
        # dataset_name으로 찾기 (fallback)
        default_project = get_or_create_default_project(db, current_user["uid"])
        dataset = db.query(Dataset).filter(
            Dataset.name == dataset_name,
            Dataset.project_id == default_project.id
        ).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found. Please provide dataset_id.")
    else:
        raise HTTPException(status_code=400, detail="Either dataset_id or dataset_name is required")
    
    # 0) 이름 중복 방지 (일반 트레이닝과 동일하게)
    if db.query(TrainingJob).filter(TrainingJob.name == job_name).first():
        raise HTTPException(status_code=400, detail="Training job with this name already exists")
    
    # S3 client 초기화 (버전 확인용)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    
    # 다음 버전 번호 가져오기
    version = _get_next_version(s3_client, settings.AWS_BUCKET_NAME, dataset_name, "ai-train")
    
    # 기본 epochs 설정 (LLM이 나중에 결정하면 업데이트됨)
    default_epochs = 20
    
    # TrainingJob 생성
    db_job = TrainingJob(
        name=job_name,
        dataset_id=dataset.id,  # dataset_id를 바로 설정
        architecture="AI-Auto",  # LLM이 자동으로 결정
        hyperparameters={
            "user_query": user_query,
            "dataset_name": dataset_name,
            "dataset_s3_prefix": dataset_s3_prefix or f"datasets/{dataset_name}/",
            "ai_mode": True,
            "version": version,
            "output_prefix": f"models/{dataset_name}/ai-train/{version}",
            "epochs": default_epochs  # 기본 epochs 추가
        },
        total_epochs=default_epochs,  # total_epochs 설정
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Model 생성 및 연결 (일반 training과 동일하게)
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    # Dataset version 생성/조회 (v0로 고정) - 이미 dataset이 확보되어 있음
    dataset_version = get_or_create_dataset_version(db, dataset.id, version_tag="v0")
    
    # Model 생성
    model = _get_or_create_model(
        db, name=job_name, project_id=default_project.id, task=None, description=f"Model for {job_name}"
    )
    
    # ModelVersion 생성
    framework_str = "AI-Auto"
    framework_version_str = getattr(settings, "FRAMEWORK_VERSION", None) or "unknown"
    mv = _get_or_create_model_v0(
        db, model_id=model.id, dataset_version_id=dataset_version.id, framework=framework_str, framework_version=framework_version_str, training_config=db_job.hyperparameters
    )
    
    # job ↔ model 연결
    db_job.model_id = model.id
    db.commit()
    db.refresh(db_job)
    
    # external_job_id 생성
    job_id_str = str(uuid.uuid4()).replace("-", "")
    db_job.hyperparameters["external_job_id"] = job_id_str
    db.commit()
    db.refresh(db_job)
    
    # 아티팩트(PT) 선점 INSERT (storage_uri만 확정)
    try:
        # 새로운 경로 구조: models/{dataset_name}/ai-train/{version}/best.pt
        object_key = f"models/{dataset_name}/ai-train/{version}/best.pt"
        
        # 중복 방지: 같은 v0에서 같은 key가 있으면 재사용
        artifact = db.query(ModelArtifact).filter(
            ModelArtifact.model_version_id == mv.id,
            ModelArtifact.storage_uri == object_key
        ).first()
        if not artifact:
            artifact = ModelArtifact(
                model_version_id=mv.id,
                storage_uri=object_key,
                format="pt"
            )
            db.add(artifact)
            db.commit()
            db.refresh(artifact)
    except Exception as e:
        logger.warning(f"[LLM Training] Failed to create artifact placeholder: {e}")
    
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
    
    logger.info(f"[LLM Training] Created training job {db_job.id} with model {model.id}")
    return db_job


def run_llm_training_pipeline(job_id: int, user_query: str, dataset_path: str):
    """
    백그라운드에서 LangGraph 기반 학습 실행
    """
    db = SessionLocal()
    job = None
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logger.error(f"[LLM Training] Job {job_id} not found")
            return
            
        logger.info(f"[LLM Training] Starting job {job_id}: {user_query}")
        
        # GPU용 UUID 생성 (RabbitMQ job_id로 사용)
        external_job_id = str(uuid.uuid4()).replace("-", "")
        
        # DB에 external_job_id 매핑 저장 (추적용)
        if not job.hyperparameters:
            job.hyperparameters = {}
        job.hyperparameters["external_job_id"] = external_job_id
        db.commit()
        
        logger.info(f"[LLM Training] Generated external_job_id: {external_job_id} for job {job_id}")
        
        # hyperparameters에서 output_prefix 가져오기 (train_trial에서 사용)
        hp = job.hyperparameters or {}
        output_prefix = hp.get("output_prefix")
        
        # LLM 기반 학습 파이프라인 실행 (UUID와 output_prefix 전달)
        builder(user_query, dataset_path, external_job_id, output_prefix)
        
        job.status = TrainingStatus.COMPLETED
        job.training_log = "AI training completed successfully"
        db.commit()
        logger.info(f"[LLM Training] Job {job_id} completed")
        
    except (asyncio.CancelledError, KeyboardInterrupt):
        # 서버 종료 시 백그라운드 태스크가 취소되는 경우 조용히 종료
        logger.info(f"[LLM Training] Job {job_id} cancelled (server shutdown)")
        if job:
            job.status = TrainingStatus.FAILED
            job.training_log = "Training cancelled due to server shutdown"
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
                job.training_log = f"AI Training failed: {str(e)}"
                db.commit()
            except Exception:
                pass  # DB 연결이 이미 끊어진 경우 무시
    finally:
        try:
            db.close()
        except Exception:
            pass  # DB 연결이 이미 끊어진 경우 무시


@router.post("/{job_id}/mark-completed", response_model=TrainingJobResponse)
async def mark_training_completed(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Manually mark a training job as completed
    Useful when GPU server completed training but didn't send train.done event
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.created_by == current_user["uid"]
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status == TrainingStatus.COMPLETED.value:
        logger.info(f"[Training] Job {job_id} is already completed")
        return job
    
    # Update status to completed
    job.status = TrainingStatus.COMPLETED
    if not job.completed_at:
        job.completed_at = datetime.utcnow()
    job.training_log = (job.training_log or "") + "\nManually marked as completed"
    
    db.commit()
    db.refresh(job)
    
    logger.info(f"[Training] Manually marked job {job_id} as completed")
    return job


@router.post("/sync-completed-status", response_model=Dict[str, Any])
async def sync_completed_status(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Check S3 for results.csv files and automatically mark training jobs as completed
    Useful for syncing status when GPU server completed but didn't send train.done event
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    
    # Get all running/pending training jobs for current user
    running_jobs = db.query(TrainingJob).filter(
        TrainingJob.created_by == current_user["uid"],
        TrainingJob.status.in_([TrainingStatus.RUNNING.value, TrainingStatus.PENDING.value])
    ).all()
    
    updated_count = 0
    updated_jobs = []
    
    for job in running_jobs:
        try:
            # 새 경로 구조에서 S3 경로 생성
            hp = job.hyperparameters or {}
            dataset_name = hp.get('dataset_name')
            version = hp.get('version')
            is_ai = hp.get('ai_mode', False)
            
            if not dataset_name or not version:
                logger.debug(f"[Training] Skipping job {job.id} - missing dataset_name or version in hyperparameters")
                continue
            
            train_type = 'ai-train' if is_ai else 'train'
            s3_key = f"models/{dataset_name}/{train_type}/{version}/results.csv"
            
            # Check if results.csv exists in S3
            try:
                s3_client.head_object(
                    Bucket=settings.AWS_BUCKET_NAME,
                    Key=s3_key
                )
                
                # results.csv exists, mark as completed
                if job.status != TrainingStatus.COMPLETED.value:
                    job.status = TrainingStatus.COMPLETED
                    if not job.completed_at:
                        job.completed_at = datetime.utcnow()
                    job.training_log = (job.training_log or "") + f"\nAuto-detected completion (results.csv found in S3: {s3_key})"
                    db.commit()
                    
                    updated_count += 1
                    updated_jobs.append({
                        "id": job.id,
                        "name": job.name,
                        "s3_path": s3_key
                    })
                    logger.info(f"[Training] Auto-marked job {job.id} ({job.name}) as completed - results.csv found at {s3_key}")
                    
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == '404':
                    # File doesn't exist, skip
                    pass
                else:
                    logger.warning(f"[Training] Error checking S3 for job {job.id}: {e}")
                    
        except Exception as e:
            logger.error(f"[Training] Error processing job {job.id}: {e}", exc_info=True)
            continue
    
    return {
        "updated_count": updated_count,
        "updated_jobs": updated_jobs,
        "message": f"Synced {updated_count} training job(s) to completed status"
    }


@router.get("/results/{model_path:path}/results.csv")
async def get_training_results_csv(
    model_path: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get training results CSV from S3
    New path structure: s3://bucket/models/{dataset_name}/{train_type}/{version}/results.csv
    - train_type: "train" or "ai-train"
    - version: "v1", "v2", "v3"...
    
    Example: model_path = "pothole/train/v1" -> s3://bucket/models/pothole/train/v1/results.csv
    """
    try:
        logger.info(f"[Training Results] Fetching results.csv for path: {model_path}")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # S3 key for results.csv (model_path already contains the full path like "pothole/train/v1")
        s3_key = f"models/{model_path}/results.csv"

        logger.info(f"[Training Results] Downloading from s3://{settings.AWS_BUCKET_NAME}/{s3_key}")

        # Download file from S3
        response = s3_client.get_object(
            Bucket=settings.AWS_BUCKET_NAME,
            Key=s3_key
        )

        # Read CSV content
        csv_content = response['Body'].read().decode('utf-8')

        logger.info(f"[Training Results] Successfully fetched results.csv ({len(csv_content)} bytes)")

        # Return as CSV response
        # Use safe filename (replace slashes with underscores)
        safe_filename = model_path.replace('/', '_')
        return Response(
            content=csv_content,
            media_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{safe_filename}_results.csv"'
            }
        )

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            logger.warning(f"[Training Results] File not found: s3://{settings.AWS_BUCKET_NAME}/{s3_key}")
            # S3에 실제로 어떤 파일들이 있는지 확인 (디버깅용)
            try:
                prefix = f"models/{model_path}/"
                list_response = s3_client.list_objects_v2(
                    Bucket=settings.AWS_BUCKET_NAME,
                    Prefix=prefix,
                    MaxKeys=10
                )
                existing_files = [obj['Key'] for obj in list_response.get('Contents', [])]
                logger.info(f"[Training Results] Files in {prefix}: {existing_files}")
            except Exception as list_err:
                logger.debug(f"[Training Results] Could not list files: {list_err}")
            
            raise HTTPException(
                status_code=404,
                detail=f"Results file not found for path '{model_path}'. Expected: s3://{settings.AWS_BUCKET_NAME}/{s3_key}. The training may not have completed yet or the file may not have been uploaded."
            )
        else:
            logger.error(f"[Training Results] S3 error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")
    except Exception as e:
        logger.error(f"[Training Results] Error fetching results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))