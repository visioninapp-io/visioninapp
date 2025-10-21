from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user_dev
from app.models.dataset import Dataset, DatasetVersion
from app.schemas.dataset_version import (
    DatasetVersionCreate, DatasetVersionUpdate, DatasetVersionResponse
)

router = APIRouter()


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
    ).order_by(DatasetVersion.version_number.desc()).all()

    return versions


@router.get("/versions/{version_id}", response_model=DatasetVersionResponse)
async def get_version(
    version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get specific dataset version"""
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    return version


@router.post("/{dataset_id}/versions", response_model=DatasetVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset_version(
    dataset_id: int,
    version_data: DatasetVersionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new dataset version with preprocessing and augmentation"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify data split sums to 1.0
    total_split = version_data.train_split + version_data.valid_split + version_data.test_split
    if abs(total_split - 1.0) > 0.001:
        raise HTTPException(
            status_code=400,
            detail=f"Data splits must sum to 1.0, got {total_split}"
        )

    # Get next version number
    last_version = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id
    ).order_by(DatasetVersion.version_number.desc()).first()

    version_number = (last_version.version_number + 1) if last_version else 1

    # Create version
    db_version = DatasetVersion(
        dataset_id=dataset_id,
        version_number=version_number,
        name=version_data.name,
        description=version_data.description,
        train_split=version_data.train_split,
        valid_split=version_data.valid_split,
        test_split=version_data.test_split,
        preprocessing_config=version_data.preprocessing_config.dict() if version_data.preprocessing_config else {},
        augmentation_config=version_data.augmentation_config.dict() if version_data.augmentation_config else {},
        status="pending",
        created_by=current_user["uid"]
    )

    db.add(db_version)
    db.commit()
    db.refresh(db_version)

    # Start background task to generate version
    background_tasks.add_task(generate_dataset_version, db_version.id, db)

    return db_version


@router.put("/versions/{version_id}", response_model=DatasetVersionResponse)
async def update_version(
    version_id: int,
    version_update: DatasetVersionUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update dataset version"""
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    update_data = version_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(version, field, value)

    db.commit()
    db.refresh(version)
    return version


@router.delete("/versions/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_version(
    version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete dataset version"""
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    db.delete(version)
    db.commit()
    return None


def generate_dataset_version(version_id: int, db: Session):
    """Background task to generate dataset version with preprocessing and augmentation"""
    from app.services.augmentation_service import AugmentationService
    from app.core.database import SessionLocal

    # Create new session for background task
    db = SessionLocal()

    try:
        version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if not version:
            return

        version.status = "generating"
        db.commit()

        # Get augmentation service
        aug_service = AugmentationService()

        # Generate version with preprocessing and augmentation
        result = aug_service.generate_version(
            version_id=version.id,
            dataset_id=version.dataset_id,
            preprocessing_config=version.preprocessing_config,
            augmentation_config=version.augmentation_config,
            train_split=version.train_split,
            valid_split=version.valid_split,
            test_split=version.test_split,
            db=db
        )

        # Update version with results
        version.total_images = result['total_images']
        version.train_images = result['train_images']
        version.valid_images = result['valid_images']
        version.test_images = result['test_images']
        version.status = "completed"
        version.generation_progress = 100

        from datetime import datetime
        version.completed_at = datetime.utcnow()

        db.commit()

    except Exception as e:
        version.status = "failed"
        db.commit()
        print(f"Error generating version: {e}")

    finally:
        db.close()
