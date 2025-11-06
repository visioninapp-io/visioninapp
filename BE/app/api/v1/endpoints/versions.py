from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user_dev
from app.models.dataset import Dataset, DatasetVersion
from app.utils.dataset_helper import (
    create_new_dataset_version,
    enrich_version_response
)
from app.schemas.dataset_version import (
    DatasetVersionCreate, 
    DatasetVersionUpdate, 
    DatasetVersionResponse
)

router = APIRouter()


@router.get("/{dataset_id}/versions/{version_id}", response_model=DatasetVersionResponse)
async def get_version(
    dataset_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get specific dataset version"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify version belongs to dataset
    version = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.dataset_id == dataset_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    return enrich_version_response(db, version)


@router.get("/{dataset_id}/versions", response_model=List[DatasetVersionResponse])
async def get_dataset_versions(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all versions of a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    versions = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id
    ).order_by(DatasetVersion.created_at.desc()).all()

    return [enrich_version_response(db, v) for v in versions]


@router.post("/{dataset_id}/versions", response_model=DatasetVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset_version(
    dataset_id: int,
    version_data: DatasetVersionCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new dataset version"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Use helper function to create version
    try:
        db_version = create_new_dataset_version(
            db=db,
            dataset_id=dataset_id,
            version_tag=version_data.version_tag  # None이면 자동 생성
        )
        
        if version_data.is_frozen:
            db_version.is_frozen = True
        
        db.commit()
        db.refresh(db_version)
        
        return enrich_version_response(db, db_version)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{dataset_id}/versions/{version_id}", response_model=DatasetVersionResponse)
async def update_version(
    dataset_id: int,
    version_id: int,
    version_update: DatasetVersionUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update dataset version"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify version belongs to dataset
    version = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.dataset_id == dataset_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    update_data = version_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(version, field, value)

    db.commit()
    db.refresh(version)
    
    return enrich_version_response(db, version)


@router.delete("/{dataset_id}/versions/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_version(
    dataset_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete dataset version"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify version belongs to dataset
    version = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.dataset_id == dataset_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    db.delete(version)
    db.commit()
    return None
