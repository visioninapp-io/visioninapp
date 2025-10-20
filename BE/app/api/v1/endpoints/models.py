from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.model import Model, ModelConversion, ModelStatus
from app.schemas.model import (
    ModelCreate, ModelUpdate, ModelResponse, ModelListResponse,
    ModelConversionRequest, ModelConversionResponse
)

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
    """Get all models"""
    query = db.query(Model)

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


@router.post("/{model_id}/download")
async def download_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Download model file"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.file_path:
        raise HTTPException(status_code=404, detail="Model file not available")

    # TODO: Implement actual file download
    # Return FileResponse or redirect to cloud storage URL

    return {
        "model_id": model.id,
        "name": model.name,
        "file_path": model.file_path,
        "download_url": f"/api/v1/models/{model_id}/file"
    }
