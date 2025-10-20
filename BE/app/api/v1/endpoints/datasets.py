from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.core.cache import cache_manager, invalidate_cache
from app.models.dataset import Dataset, Image, Annotation, DatasetStatus
from app.schemas.dataset import (
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetStats,
    ImageResponse, AnnotationCreate, AnnotationResponse, AutoAnnotationRequest
)
from app.utils.file_storage import file_storage

router = APIRouter()


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get overall dataset statistics"""
    total_images = db.query(Image).count()
    total_datasets = db.query(Dataset).count()

    datasets = db.query(Dataset).all()
    total_classes = sum(d.total_classes for d in datasets)

    annotated_images = db.query(Image).filter(Image.is_annotated == 1).count()
    auto_annotation_rate = (annotated_images / total_images * 100) if total_images > 0 else 0

    return {
        "total_images": total_images,
        "total_datasets": total_datasets,
        "total_classes": total_classes,
        "auto_annotation_rate": auto_annotation_rate
    }


@router.get("/", response_model=List[DatasetResponse])
async def get_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all datasets with caching"""
    cache_key = f"datasets:list:{current_user['uid']}:{skip}:{limit}"

    # Try to get from cache
    cached_data = cache_manager.get(cache_key)
    if cached_data is not None:
        print(f"✓ Cache hit: {cache_key}")
        return cached_data

    # Query database
    print(f"⊗ Cache miss: {cache_key}")
    datasets = db.query(Dataset).offset(skip).limit(limit).all()

    # Convert to dict for serialization
    datasets_dict = [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "status": d.status.value if hasattr(d.status, 'value') else d.status,
            "total_images": d.total_images,
            "annotated_images": d.annotated_images,
            "total_classes": d.total_classes,
            "class_names": d.class_names,
            "auto_annotation_enabled": bool(d.auto_annotation_enabled),
            "auto_annotation_model_id": d.auto_annotation_model_id,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            "created_by": d.created_by
        }
        for d in datasets
    ]

    # Cache the result
    cache_manager.set(cache_key, datasets_dict, ttl=60)  # Cache for 1 minute

    return datasets


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new dataset"""
    db_dataset = db.query(Dataset).filter(Dataset.name == dataset.name).first()
    if db_dataset:
        raise HTTPException(status_code=400, detail="Dataset with this name already exists")

    db_dataset = Dataset(
        **dataset.dict(),
        created_by=current_user["uid"]
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    # Invalidate datasets cache
    invalidate_cache("datasets")
    print("✓ Cache invalidated: datasets")

    return db_dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update a dataset"""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = dataset_update.dict(exclude_unset=True)

    # Convert boolean to int for SQLite
    if "auto_annotation_enabled" in update_data:
        update_data["auto_annotation_enabled"] = int(update_data["auto_annotation_enabled"])

    for field, value in update_data.items():
        setattr(db_dataset, field, value)

    db.commit()
    db.refresh(db_dataset)
    return db_dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()
    return None


@router.get("/{dataset_id}/images", response_model=List[ImageResponse])
async def get_dataset_images(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all images in a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    images = db.query(Image).filter(Image.dataset_id == dataset_id).offset(skip).limit(limit).all()
    return images


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    dataset_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Upload images to create a new dataset or add to existing dataset"""

    # Validate files list
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # If dataset_id provided, use existing dataset
    if dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        # Create new dataset
        if not name:
            raise HTTPException(status_code=400, detail="Dataset name is required for new dataset")

        existing = db.query(Dataset).filter(Dataset.name == name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")

        dataset = Dataset(
            name=name,
            description=description or "",
            total_images=0,
            total_classes=0,
            status=DatasetStatus.UPLOADING,
            created_by=current_user["uid"]
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

    # Save files to disk and create Image records
    try:
        dataset.status = DatasetStatus.UPLOADING
        db.commit()

        successful_uploads, failed_uploads = await file_storage.save_images_batch(files, dataset.id)

        # Create Image records for successful uploads
        for upload in successful_uploads:
            metadata = upload["metadata"]
            db_image = Image(
                dataset_id=dataset.id,
                filename=metadata["stored_filename"],
                file_path=metadata["relative_path"],
                file_size=metadata["file_size"],
                width=metadata.get("width"),
                height=metadata.get("height"),
                is_annotated=0
            )
            db.add(db_image)

        # Update dataset counts
        dataset.total_images += len(successful_uploads)
        dataset.status = DatasetStatus.READY if len(failed_uploads) == 0 else DatasetStatus.PROCESSING
        db.commit()
        db.refresh(dataset)

        # Invalidate datasets cache
        invalidate_cache("datasets")
        print("✓ Cache invalidated after upload: datasets")

        response = {
            "message": f"Successfully uploaded {len(successful_uploads)} images",
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "total_images": dataset.total_images,
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "dataset": dataset
        }

        if failed_uploads:
            response["failed_files"] = failed_uploads

        return response

    except Exception as e:
        dataset.status = DatasetStatus.ERROR
        db.commit()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/{dataset_id}/images/upload", status_code=status.HTTP_201_CREATED)
async def upload_images_to_dataset(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Upload images to a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Update dataset status
        original_status = dataset.status
        dataset.status = DatasetStatus.UPLOADING
        db.commit()

        # Save images to disk
        successful_uploads, failed_uploads = await file_storage.save_images_batch(files, dataset.id)

        # Create Image records
        for upload in successful_uploads:
            metadata = upload["metadata"]
            db_image = Image(
                dataset_id=dataset.id,
                filename=metadata["stored_filename"],
                file_path=metadata["relative_path"],
                file_size=metadata["file_size"],
                width=metadata.get("width"),
                height=metadata.get("height"),
                is_annotated=0
            )
            db.add(db_image)

        # Update dataset
        dataset.total_images += len(successful_uploads)
        dataset.status = original_status if len(failed_uploads) == 0 else DatasetStatus.PROCESSING
        db.commit()

        response = {
            "message": f"Successfully uploaded {len(successful_uploads)} images",
            "dataset_id": dataset_id,
            "total_images": dataset.total_images,
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads)
        }

        if failed_uploads:
            response["failed_files"] = failed_uploads

        return response

    except Exception as e:
        dataset.status = DatasetStatus.ERROR
        db.commit()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/annotations", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
async def create_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new annotation"""
    image = db.query(Image).filter(Image.id == annotation.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    db_annotation = Annotation(
        **annotation.dict(exclude={"is_auto_generated"}),
        is_auto_generated=int(annotation.is_auto_generated)
    )
    db.add(db_annotation)

    # Mark image as annotated
    image.is_annotated = 1

    db.commit()
    db.refresh(db_annotation)
    return db_annotation


@router.post("/auto-annotate")
async def auto_annotate_dataset(
    request: AutoAnnotationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Trigger auto-annotation for a dataset using YOLO model"""
    from app.services.auto_annotation_service import get_auto_annotation_service
    from app.utils.file_storage import file_storage
    from pathlib import Path

    # 데이터셋 확인
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 데이터셋에 이미지가 있는지 확인
    images = db.query(Image).filter(Image.dataset_id == request.dataset_id).all()
    if not images:
        raise HTTPException(status_code=400, detail="No images in dataset")

    # 상태 업데이트
    dataset.status = DatasetStatus.PROCESSING
    dataset.auto_annotation_enabled = 1
    dataset.auto_annotation_model_id = request.model_id
    db.commit()

    try:
        # 자동 어노테이션 서비스 로드
        service = get_auto_annotation_service()

        # 모델 로드
        if not service.load_model():
            raise HTTPException(
                status_code=500,
                detail="Failed to load model. Please train the model first (run AI/scripts/train_yolo.py)"
            )

        # Confidence threshold (요청에서 받거나 기본값 0.25)
        conf_threshold = getattr(request, 'confidence_threshold', 0.25)

        # 각 이미지에 대해 자동 어노테이션 실행
        total_annotations = 0
        annotated_images_count = 0
        base_upload_dir = Path("uploads")

        for img in images:
            img_path = base_upload_dir / img.file_path

            if not img_path.exists():
                continue

            # 추론 실행
            annotations = service.predict_image(str(img_path), conf_threshold)

            if annotations:
                # 어노테이션 저장
                for ann in annotations:
                    bbox = ann['bbox']
                    db_annotation = Annotation(
                        image_id=img.id,
                        class_id=ann['class_id'],
                        class_name=ann['class_name'],
                        x_center=bbox['x_center'],
                        y_center=bbox['y_center'],
                        width=bbox['width'],
                        height=bbox['height'],
                        confidence=ann['confidence'],
                        is_auto_generated=1,
                        is_verified=0
                    )
                    db.add(db_annotation)
                    total_annotations += 1

                # 이미지를 어노테이션됨으로 표시
                img.is_annotated = 1
                annotated_images_count += 1

        # 커밋
        db.commit()

        # 데이터셋 통계 업데이트
        dataset.annotated_images = annotated_images_count
        dataset.status = DatasetStatus.ANNOTATED if annotated_images_count > 0 else DatasetStatus.READY

        # 클래스 이름 업데이트
        model_info = service.get_model_info()
        if model_info['class_names']:
            dataset.class_names = model_info['class_names']
            dataset.total_classes = len(model_info['class_names'])

        db.commit()

        return {
            "message": "Auto-annotation completed",
            "dataset_id": request.dataset_id,
            "model_id": request.model_id,
            "status": "completed",
            "total_images": len(images),
            "annotated_images": annotated_images_count,
            "total_annotations": total_annotations,
            "confidence_threshold": conf_threshold
        }

    except Exception as e:
        # 에러 발생 시 상태 복구
        dataset.status = DatasetStatus.ERROR
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Auto-annotation failed: {str(e)}"
        )
