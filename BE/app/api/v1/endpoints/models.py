from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.model import Model, ModelConversion, ModelStatus
from app.models.model_artifact import ModelArtifact
from app.schemas.model import (
    ModelCreate, ModelUpdate, ModelResponse, ModelListResponse,
    ModelConversionRequest, ModelConversionResponse
)
from app.schemas.model_artifact import (
    ModelArtifactResponse, ModelUploadRequest, ModelUploadCompleteRequest
)
from app.services.presigned_url_service import presigned_url_service

router = APIRouter()


@router.get("/", response_model=List[ModelListResponse])
async def get_models(
    skip: int = 0,
    limit: int = 100,
    framework: str = None,
    status_filter: str = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all models for current user"""
    query = db.query(Model).filter(Model.created_by == current_user["uid"])

    if framework:
        query = query.filter(Model.framework == framework)

    if status_filter:
        query = query.filter(Model.status == status_filter)

    models = query.offset(skip).limit(limit).all()
    return models


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new model"""
    existing = db.query(Model).filter(Model.name == model.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this name already exists")

    db_model = Model(
        **model.dict(),
        created_by=current_user["uid"]
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model)
    db.commit()
    return None


@router.post("/convert", response_model=ModelConversionResponse, status_code=status.HTTP_201_CREATED)
async def convert_model(
    request: ModelConversionRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Convert model to different format (ONNX, TensorRT, etc.)"""
    model = db.query(Model).filter(Model.id == request.source_model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create conversion record
    conversion = ModelConversion(
        source_model_id=request.source_model_id,
        target_framework=request.target_framework,
        optimization_level=request.optimization_level,
        precision=request.precision,
        status="pending"
    )
    db.add(conversion)
    db.commit()
    db.refresh(conversion)

    # TODO: Implement actual model conversion (async task)
    # This would use tools like:
    # - PyTorch -> ONNX: torch.onnx.export()
    # - ONNX -> TensorRT: trtexec
    # - TensorFlow -> TFLite, etc.

    return conversion


@router.get("/convert/{conversion_id}", response_model=ModelConversionResponse)
async def get_conversion_status(
    conversion_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get model conversion status"""
    conversion = db.query(ModelConversion).filter(ModelConversion.id == conversion_id).first()
    if not conversion:
        raise HTTPException(status_code=404, detail="Conversion not found")
    return conversion


@router.get("/{model_id}/conversions", response_model=List[ModelConversionResponse])
async def get_model_conversions(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all conversions for a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    conversions = db.query(ModelConversion).filter(
        ModelConversion.source_model_id == model_id
    ).all()

    return conversions


@router.post("/{model_id}/presigned-upload")
async def generate_model_upload_url(
    model_id: int,
    request: ModelUploadRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
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
    # 모델 존재 및 권한 확인
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.created_by == current_user["uid"]
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
    current_user: dict = Depends(get_current_user_dev)
):
    """
    모델 파일 S3 업로드 완료 알림 - ModelArtifact에 메타데이터 저장
    
    - 업로드된 파일 정보를 ModelArtifact 테이블에 기록
    - is_primary=True인 경우 기존 primary를 false로 변경
    """
    # 모델 존재 및 권한 확인
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.created_by == current_user["uid"]
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # S3 파일 존재 확인
        if not presigned_url_service.check_object_exists(request.s3_key):
            raise HTTPException(status_code=404, detail="File not found in S3")
        
        # is_primary인 경우 기존 primary를 false로 변경
        if request.is_primary:
            db.query(ModelArtifact).filter(
                ModelArtifact.model_id == model_id,
                ModelArtifact.is_primary == True
            ).update({"is_primary": False})
        
        # ModelArtifact에 메타데이터 저장
        artifact = ModelArtifact(
            model_id=model_id,
            kind=request.kind,
            version=request.version,
            s3_key=request.s3_key,
            file_size=request.file_size,
            checksum=request.checksum,
            is_primary=request.is_primary,
            created_by=current_user["uid"]
        )
        
        db.add(artifact)
        
        # Model 테이블 업데이트 (primary artifact인 경우)
        if request.is_primary:
            model.file_path = request.s3_key
            model.file_size = request.file_size
            if model.status == ModelStatus.TRAINING:
                model.status = ModelStatus.COMPLETED
        
        db.commit()
        db.refresh(artifact)
        
        return {
            "success": True,
            "artifact_id": artifact.id,
            "model_id": model_id,
            "s3_key": artifact.s3_key,
            "is_primary": artifact.is_primary
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
    current_user: dict = Depends(get_current_user_dev)
):
    """
    모델의 artifacts 목록 조회 (PT, ONNX, TRT 등)
    """
    # 모델 존재 확인
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Artifacts 조회
    artifacts = db.query(ModelArtifact).filter(
        ModelArtifact.model_id == model_id
    ).order_by(ModelArtifact.is_primary.desc(), ModelArtifact.created_at.desc()).all()
    
    return artifacts


@router.get("/artifacts/{artifact_id}/presigned-download")
async def get_artifact_download_url(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
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
    
    # 모델 권한 확인
    model = db.query(Model).filter(
        Model.id == artifact.model_id,
        Model.created_by == current_user["uid"]
    ).first()
    
    if not model:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # 원본 파일명 생성 (모델명_버전.확장자)
        file_ext = artifact.s3_key.split('.')[-1] if '.' in artifact.s3_key else artifact.kind
        download_filename = f"{model.name}_v{artifact.version}.{file_ext}"
        
        # Presigned GET URL 생성
        url_data = presigned_url_service.generate_download_url(
            s3_key=artifact.s3_key,
            expiration=3600,
            filename=download_filename
        )
        
        return {
            "success": True,
            "artifact_id": artifact.id,
            "model_id": artifact.model_id,
            "kind": artifact.kind,
            "version": artifact.version,
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
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Download model file (primary artifact)
    
    Legacy endpoint - redirects to primary artifact download
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # primary artifact 찾기
    primary_artifact = db.query(ModelArtifact).filter(
        ModelArtifact.model_id == model_id,
        ModelArtifact.is_primary == True
    ).first()
    
    if not primary_artifact:
        raise HTTPException(status_code=404, detail="Model file not available")
    
    # Presigned URL 생성
    try:
        file_ext = primary_artifact.s3_key.split('.')[-1] if '.' in primary_artifact.s3_key else 'pt'
        download_filename = f"{model.name}.{file_ext}"
        
        url_data = presigned_url_service.generate_download_url(
            s3_key=primary_artifact.s3_key,
            expiration=3600,
            filename=download_filename
        )
        
        return {
            "model_id": model.id,
            "name": model.name,
            "file_path": primary_artifact.s3_key,
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
    current_user: dict = Depends(get_current_user_dev)
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

    # Check if model file exists
    if not model.file_path:
        raise HTTPException(status_code=404, detail="Model file not available")

    model_path = Path(model.file_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found at {model.file_path}")

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
