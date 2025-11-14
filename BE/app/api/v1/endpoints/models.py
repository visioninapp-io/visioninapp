from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
from datetime import datetime
import os
import boto3
import hashlib
from botocore.exceptions import ClientError
from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.config import settings
from app.models.model import Model
from app.utils.project_helper import get_or_create_default_project
from app.models.model_artifact import ModelArtifact
from app.schemas.model import (
    ModelCreate, ModelUpdate, ModelResponse, ModelListResponse
)
from app.schemas.model_artifact import (
    ModelArtifactResponse, ModelUploadRequest, ModelUploadCompleteRequest
)
from app.services.presigned_url_service import presigned_url_service

router = APIRouter()


@router.get("/", response_model=List[ModelListResponse])
async def get_models(
    skip: int = 0,
    limit: int = 10000,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all models for current user"""
    # created_by 필드가 없으므로 project 기반으로 필터링
    from app.utils.project_helper import get_or_create_default_project
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    query = db.query(Model).filter(Model.project_id == default_project.id)
    models = query.offset(skip).limit(limit).all()
    return models


@router.get("/trained", response_model=List[dict])
async def get_trained_models(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get trained models from database (user's models only), enriched with S3 metadata"""
    try:
        # Get user's default project
        from app.models.model_version import ModelVersion
        
        default_project = get_or_create_default_project(db, current_user["uid"])
        
        # Query models from database that belong to user's project
        user_models = db.query(Model).filter(
            Model.project_id == default_project.id
        ).all()
        
        if not user_models:
            return []
        
        # Get S3 client for metadata (optional - only if S3 is configured)
        s3_client = None
        if (settings.AWS_BUCKET_NAME and 
            settings.AWS_ACCESS_KEY_ID and 
            settings.AWS_SECRET_ACCESS_KEY):
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
            except Exception:
                s3_client = None  # S3 not available, continue without metadata
        
        trained_models = []
        
        # For each model, get its artifacts (PT files)
        for model in user_models:
            # Get model versions
            model_versions = db.query(ModelVersion).filter(
                ModelVersion.model_id == model.id
            ).all()
            
            for model_version in model_versions:
                # Get PT artifacts for this model version
                pt_artifacts = db.query(ModelArtifact).filter(
                    ModelArtifact.model_version_id == model_version.id,
                    ModelArtifact.format == 'pt'  # Only PT files
                ).all()
                
                for artifact in pt_artifacts:
                    if not artifact.storage_uri:
                        continue
                    
                    # Get S3 metadata if available
                    file_size_bytes = None
                    file_size_mb = None
                    modified_timestamp = None
                    
                    if s3_client and artifact.storage_uri:
                        try:
                            # Try to get object metadata from S3
                            response = s3_client.head_object(
                                Bucket=settings.AWS_BUCKET_NAME,
                                Key=artifact.storage_uri
                            )
                            file_size_bytes = response.get('ContentLength', 0)
                            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                            modified_time = response.get('LastModified', None)
                            modified_timestamp = modified_time.timestamp() if modified_time else None
                        except ClientError:
                            # File doesn't exist in S3, but artifact exists in DB
                            pass
                    
                    # Fallback to artifact metadata if S3 metadata not available
                    if file_size_bytes is None:
                        file_size_bytes = artifact.size_bytes
                        if file_size_bytes:
                            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                    
                    if modified_timestamp is None and artifact.created_at:
                        modified_timestamp = artifact.created_at.timestamp()
                    
                    # Extract filename from storage_uri
                    filename = artifact.storage_uri.split('/')[-1] if artifact.storage_uri else None
                    
                    # Generate unique ID (use artifact ID or model ID)
                    unique_id = artifact.id if artifact.id else int(hashlib.md5(f"{model.id}_{artifact.storage_uri}".encode()).hexdigest()[:8], 16) % 1000000
                    
                    model_data = {
                        "id": unique_id,
                        "model_id": model.id,
                        "model_name": model.name,
                        "model_path": f"s3://{settings.AWS_BUCKET_NAME}/{artifact.storage_uri}" if artifact.storage_uri else None,
                        "relative_path": artifact.storage_uri,
                        "s3_key": artifact.storage_uri,
                        "s3_bucket": settings.AWS_BUCKET_NAME,
                        "file_size_mb": file_size_mb,
                        "file_size_bytes": file_size_bytes,
                        "modified_at": modified_timestamp,
                        "filename": filename,
                        "has_artifact": True,  # Always true since we're querying from DB
                        "artifact_id": artifact.id,
                        "model_version_id": model_version.id
                    }
                    
                    trained_models.append(model_data)
        
        # Sort by model_id descending (newest first), or by modified_at if no model_id
        trained_models.sort(
            key=lambda x: (
                x['model_id'] if x['model_id'] is not None else 0,
                x['modified_at'] if x['modified_at'] else 0
            ),
            reverse=True
        )
        
        return trained_models
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list trained models: {str(e)}"
        )


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new model"""
    existing = db.query(Model).filter(Model.name == model.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this name already exists")

    # Get or create default project for user
    default_project = get_or_create_default_project(db, current_user["uid"])

    db_model = Model(
        **model.dict(),
        project_id=default_project.id
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a model"""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    update_data = model_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_model, field, value)

    db.commit()
    db.refresh(db_model)
    return db_model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a model, its training job, and all artifacts from DB and S3"""
    from app.models.model_version import ModelVersion
    from app.models.training import TrainingJob
    import logging

    logger = logging.getLogger(__name__)

    # Check if model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # Get all model versions and their artifacts
        model_versions = db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).all()

        # Delete all artifacts from S3
        for version in model_versions:
            for artifact in version.artifacts:
                if artifact.storage_uri:
                    try:
                        logger.info(f"[Model Delete] Deleting S3 object: {artifact.storage_uri}")
                        s3_client.delete_object(
                            Bucket=settings.AWS_BUCKET_NAME,
                            Key=artifact.storage_uri
                        )
                        logger.info(f"[Model Delete] Successfully deleted: {artifact.storage_uri}")
                    except Exception as e:
                        logger.error(f"[Model Delete] Failed to delete S3 object {artifact.storage_uri}: {e}")
                        # Continue even if S3 deletion fails

        # Delete associated training job if exists
        training_job = db.query(TrainingJob).filter(TrainingJob.model_id == model_id).first()
        if training_job:
            logger.info(f"[Model Delete] Deleting associated training job {training_job.id}")
            db.delete(training_job)

        # Delete from database (cascade will handle versions and artifacts)
        db.delete(model)
        db.commit()

        logger.info(f"[Model Delete] Successfully deleted model {model_id} and training job from DB and S3")
        return None

    except Exception as e:
        db.rollback()
        logger.error(f"[Model Delete] Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/{model_id}/presigned-upload")
async def generate_model_upload_url(
    model_id: int,
    request: ModelUploadRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    모델 파일 업로드용 Presigned URL 발급
    
    - PT, ONNX, TRT 등 모델 파일 업로드
    - 클라이언트가 받은 URL로 S3에 직접 업로드
    - URL 만료 시간: 30분
    
    사용 후:
    1. S3에 파일 업로드 완료
    2. POST /api/v1/models/{model_id}/upload-complete 호출
    """
    # 모델 존재 및 권한 확인 (project로 확인)
    from app.utils.project_helper import get_or_create_default_project
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.project_id == default_project.id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Presigned URL 생성
        url_data = presigned_url_service.generate_model_upload_url(
            model_id=model_id,
            filename=request.filename,
            content_type=request.content_type,
            expiration=1800  # 30분
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "upload_url": url_data["upload_url"],
            "s3_key": url_data["s3_key"],
            "unique_filename": url_data["unique_filename"],
            "original_filename": url_data["original_filename"],
            "expires_in": url_data["expires_in"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/{model_id}/upload-complete")
async def confirm_model_upload_complete(
    model_id: int,
    request: ModelUploadCompleteRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    모델 파일 S3 업로드 완료 알림 - ModelArtifact에 메타데이터 저장
    
    - 업로드된 파일 정보를 ModelArtifact 테이블에 기록
    - is_primary=True인 경우 기존 primary를 false로 변경
    """
    # 모델 존재 및 권한 확인 (project로 확인)
    from app.utils.project_helper import get_or_create_default_project
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.project_id == default_project.id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # S3 파일 존재 확인
        if not presigned_url_service.check_object_exists(request.s3_key):
            raise HTTPException(status_code=404, detail="File not found in S3")
        
        # ModelVersion 필요 (임시로 첫 번째 버전 사용)
        from app.models.model_version import ModelVersion
        model_version = db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).first()
        
        if not model_version:
            raise HTTPException(status_code=404, detail="Model version not found. Please create a model version first.")
        
        # ModelArtifact에 메타데이터 저장
        # 주의: 실제로는 format, device, precision, storage_uri, sha256, size_bytes 등이 필요
        # 현재는 임시로 호환성을 위해 기본값 사용
        artifact = ModelArtifact(
            model_version_id=model_version.id,
            format=request.kind or "pt",  # kind -> format 매핑
            device="cpu",  # 기본값
            precision=1.0,  # 기본값
            storage_uri=request.s3_key,
            sha256=request.checksum or "",  # checksum이 없으면 빈 문자열
            size_bytes=request.file_size
        )
        
        db.add(artifact)
        
        db.commit()
        db.refresh(artifact)
        
        return {
            "success": True,
            "artifact_id": artifact.id,
            "model_version_id": model_version.id,
            "storage_uri": artifact.storage_uri
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process upload completion: {str(e)}"
        )


@router.get("/{model_id}/artifacts", response_model=List[ModelArtifactResponse])
async def get_model_artifacts(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    모델의 artifacts 목록 조회 (S3에서 직접 조회)
    S3 구조: models/{model_name}/{model_name}.pt 및 models/{model_name}/results.csv
    """
    # 모델 존재 확인
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if S3 is configured
    if not settings.AWS_BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail="AWS_BUCKET_NAME not configured. Cannot list artifacts from S3."
        )
    
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        raise HTTPException(
            status_code=500,
            detail="AWS credentials not configured. Cannot access S3."
        )
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        
        # Try to find model folder in S3 (format: model_{id} or use model name)
        model_folder = f"model_{model_id}"
        prefix = f"models/{model_folder}/"
        
        artifacts = []
        
        # List objects in the model folder
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=settings.AWS_BUCKET_NAME, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    
                    # Only include .pt and .csv files
                    if not (key.endswith('.pt') or key.endswith('.csv')):
                        continue
                    
                    # Determine file kind
                    if key.endswith('.pt'):
                        kind = 'pt'
                    elif key.endswith('.csv'):
                        kind = 'csv'
                    else:
                        continue
                    
                    # Get file metadata
                    file_size_bytes = obj.get('Size', 0)
                    modified_time = obj.get('LastModified', None)
                    
                    # Create artifact response
                    artifact = ModelArtifactResponse(
                        id=len(artifacts) + 1,  # Temporary ID for response
                        model_id=model_id,
                        kind=kind,
                        version="1.0",
                        s3_key=key,
                        file_size=file_size_bytes,
                        checksum=None,
                        is_primary=(kind == 'pt'),  # .pt files are primary
                        created_at=modified_time if modified_time else datetime.now(),
                        created_by=current_user.get("uid", "system")
                    )
                    artifacts.append(artifact)
        
        except ClientError as e:
            # If model_{id} folder doesn't exist, return empty list
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                return []
            raise
        
        return artifacts
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list artifacts from S3: {error_code} - {error_message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list model artifacts: {str(e)}"
        )


@router.get("/artifacts/{artifact_id}/presigned-download")
async def get_artifact_download_url(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    ModelArtifact 다운로드용 Presigned URL 발급
    
    - PT, ONNX, TRT 등 모델 파일 다운로드
    - URL 만료 시간: 1시간
    """
    # Artifact 조회
    artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # 모델 권한 확인 (model_version을 통해 확인)
    from app.models.model_version import ModelVersion
    from app.utils.project_helper import get_or_create_default_project
    
    model_version = db.query(ModelVersion).filter(
        ModelVersion.id == artifact.model_version_id
    ).first()
    
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    default_project = get_or_create_default_project(db, current_user["uid"])
    if model_version.model.project_id != default_project.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # 원본 파일명 생성 (모델명_버전.확장자)
        file_ext = artifact.storage_uri.split('.')[-1] if '.' in artifact.storage_uri else artifact.format
        download_filename = f"{model_version.model.name}_{model_version.version_tag}.{file_ext}"
        
        # Presigned GET URL 생성
        url_data = presigned_url_service.generate_download_url(
            s3_key=artifact.storage_uri,
            expiration=3600,
            filename=download_filename
        )
        
        return {
            "success": True,
            "artifact_id": artifact.id,
            "model_version_id": artifact.model_version_id,
            "format": artifact.format,
            "download_url": url_data["download_url"],
            "download_filename": download_filename,
            "expires_in": url_data["expires_in"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )


@router.post("/{model_id}/download")
async def download_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Download model file (primary artifact)
    
    Legacy endpoint - redirects to primary artifact download
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # ModelVersion의 첫 번째 artifact 찾기
    from app.models.model_version import ModelVersion
    model_version = db.query(ModelVersion).filter(
        ModelVersion.model_id == model_id
    ).order_by(ModelVersion.id.desc()).first()
    
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    artifact = db.query(ModelArtifact).filter(
        ModelArtifact.model_version_id == model_version.id
    ).order_by(ModelArtifact.created_at.desc()).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Model file not available")
    
    # Presigned URL 생성
    try:
        file_ext = artifact.storage_uri.split('.')[-1] if '.' in artifact.storage_uri else artifact.format
        download_filename = f"{model.name}.{file_ext}"
        
        url_data = presigned_url_service.generate_download_url(
            s3_key=artifact.storage_uri,
            expiration=3600,
            filename=download_filename
        )
        
        return {
            "model_id": model.id,
            "name": model.name,
            "storage_uri": artifact.storage_uri,
            "download_url": url_data["download_url"],
            "expires_in": url_data["expires_in"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )


@router.post("/{model_id}/predict")
async def predict_with_model(
    model_id: int,
    file: UploadFile = File(...),
    confidence: float = 0.25,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Run inference with a .pt model file"""
    from pathlib import Path
    from app.services.auto_annotation_service import get_auto_annotation_service
    import tempfile
    import shutil

    # Get model from database
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # ModelVersion의 artifact에서 파일 경로 찾기
    from app.models.model_version import ModelVersion
    model_version = db.query(ModelVersion).filter(
        ModelVersion.model_id == model_id
    ).order_by(ModelVersion.id.desc()).first()
    
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    artifact = db.query(ModelArtifact).filter(
        ModelArtifact.model_version_id == model_version.id
    ).order_by(ModelArtifact.created_at.desc()).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Model file not available")

    model_path = Path(artifact.storage_uri)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found at {artifact.storage_uri}")

    # Save uploaded image to temporary file
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Load model and run inference
        service = get_auto_annotation_service()

        # Load the specific model
        if not service.load_model(str(model_path)):
            raise HTTPException(status_code=500, detail="Failed to load model")

        # Run prediction
        predictions = service.predict_image(tmp_path, confidence)

        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)

        # Get model info
        model_info = service.get_model_info()

        return {
            "model_id": model.id,
            "model_name": model.name,
            "predictions": predictions,
            "num_detections": len(predictions),
            "confidence_threshold": confidence,
            "class_names": model_info.get("class_names", [])
        }

    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")