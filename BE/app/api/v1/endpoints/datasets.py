from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import io
import zipfile
import base64
from datetime import datetime
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.core.cache import cache_manager, invalidate_cache
from app.models.dataset import Dataset, Image, Annotation, DatasetStatus
from app.schemas.dataset import (
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetStats,
    ImageResponse, AnnotationCreate, AnnotationResponse, AutoAnnotationRequest
)
from app.utils.file_storage import file_storage
from app.services.dataset_cache_service import dataset_cache_service

router = APIRouter()


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get overall dataset statistics for current user"""
    # Get user's datasets
    user_datasets = db.query(Dataset).filter(Dataset.created_by == current_user["uid"]).all()
    dataset_ids = [d.id for d in user_datasets]

    # Count images only from user's datasets
    total_images = db.query(Image).filter(Image.dataset_id.in_(dataset_ids)).count() if dataset_ids else 0
    total_datasets = len(user_datasets)
    total_classes = sum(d.total_classes for d in user_datasets)

    annotated_images = db.query(Image).filter(
        Image.dataset_id.in_(dataset_ids),
        Image.is_annotated == 1
    ).count() if dataset_ids else 0

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
        print(f"[CACHE HIT] {cache_key}")
        return cached_data

    # Query database - Filter by current user
    print(f"[CACHE MISS] {cache_key}")
    datasets = db.query(Dataset).filter(Dataset.created_by == current_user["uid"]).offset(skip).limit(limit).all()

    # Convert to dict for serialization
    datasets_dict = [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "dataset_type": d.dataset_type.value if hasattr(d.dataset_type, 'value') else (d.dataset_type or "object_detection"),
            "status": d.status.value if hasattr(d.status, 'value') else d.status,
            "total_images": d.total_images,
            "annotated_images": d.annotated_images,
            "total_classes": d.total_classes,
            "class_names": d.class_names or [],
            "class_colors": d.class_colors or {},
            "auto_annotation_enabled": bool(d.auto_annotation_enabled),
            "auto_annotation_model_id": d.auto_annotation_model_id,
            "is_public": bool(d.is_public),
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            "created_by": d.created_by
        }
        for d in datasets
    ]

    # Cache the result
    cache_manager.set(cache_key, datasets_dict, ttl=60)  # Cache for 1 minute

    return datasets_dict


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

    # Get dataset data and ensure default values
    dataset_data = dataset.dict()
    if dataset_data.get("class_names") is None:
        dataset_data["class_names"] = []
    if dataset_data.get("class_colors") is None:
        dataset_data["class_colors"] = {}

    db_dataset = Dataset(
        **dataset_data,
        created_by=current_user["uid"]
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    # Invalidate datasets cache
    invalidate_cache("datasets")
    print("[CACHE] Invalidated: datasets")

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

    print(f"[DELETE DATASET] Deleting dataset {dataset_id}: {dataset.name}")
    
    # Invalidate dataset images cache before deletion
    dataset_cache_service.invalidate_dataset_cache(dataset_id)
    print(f"[CACHE] Invalidated images cache for dataset {dataset_id}")
    
    db.delete(dataset)
    db.commit()

    # Invalidate cache to ensure frontend gets updated list
    invalidate_cache("datasets")
    print("[CACHE] Invalidated after dataset deletion: datasets")

    return None


@router.get("/{dataset_id}/images")
async def get_dataset_images(
    dataset_id: int,
    page: int = 1,
    limit: int = 1000,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Get images in a dataset with base64 encoding and Redis caching
    
    Args:
        dataset_id: Dataset ID
        page: Page number (1-based, default: 1)
        limit: Number of images per page (default: 50)
    
    Returns:
        Dictionary with base64 encoded images and metadata
        
    Cache Strategy:
        - First request: Loads from S3/local storage, encodes to base64, caches in Redis
        - Subsequent requests: Returns cached data from Redis (very fast)
        - Cache is automatically invalidated on dataset updates
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all images for this dataset
    all_images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
    
    if not all_images:
        return {
            "dataset_id": dataset_id,
            "page": page,
            "page_size": limit,
            "cached": False,
            "images": [],
            "total_images": 0,
            "total_pages": 0
        }
    
    print(f"[GET IMAGES] Dataset {dataset_id}: {len(all_images)} total images, page {page}, limit {limit}")
    
    # Use cache service to get base64 encoded images
    try:
        result = await dataset_cache_service.get_cached_images(
            dataset_id=dataset_id,
            images_db=all_images,
            page=page,
            limit=limit
        )
        
        # Add total pages
        result["total_pages"] = (result["total_images"] + limit - 1) // limit
        
        print(f"[GET IMAGES] Returning {len(result['images'])} images (cached: {result['cached']})")
        
        return result
    except Exception as e:
        print(f"[ERROR] Failed to get images for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load images from S3: {str(e)}"
        )


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

    # Upload files to S3 and create Image records
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

        # Invalidate datasets cache and dataset images cache
        invalidate_cache("datasets")
        print("[CACHE] Invalidated after upload: datasets")
        
        dataset_cache_service.invalidate_dataset_cache(dataset.id)
        print(f"[CACHE] Invalidated images cache for dataset {dataset.id}")

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

        # Upload images to S3
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
        
        # Invalidate dataset images cache
        dataset_cache_service.invalidate_dataset_cache(dataset_id)
        print(f"[CACHE] Invalidated images cache for dataset {dataset_id}")

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


@router.get("/images/{image_id}/annotations", response_model=List[AnnotationResponse])
async def get_image_annotations(
    image_id: int,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all annotations for a specific image, optionally filtered by minimum confidence threshold"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get all annotations for the image
    query = db.query(Annotation).filter(Annotation.image_id == image_id)

    # Apply confidence threshold filter if specified
    if min_confidence is not None:
        query = query.filter(Annotation.confidence >= min_confidence)
        print(f"[GET ANNOTATIONS] Filtering by confidence >= {min_confidence}")

    annotations = query.all()
    print(f"[GET ANNOTATIONS] Found {len(annotations)} annotations for image {image_id}")

    return annotations


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
    print(f"[AUTO-ANNOTATE] Received request: dataset_id={request.dataset_id}, model_id={request.model_id}, conf={getattr(request, 'confidence_threshold', 0.25)}")

    from app.services.auto_annotation_service import get_auto_annotation_service
    from app.utils.file_storage import file_storage
    from pathlib import Path

    # 데이터셋 확인
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        print(f"[AUTO-ANNOTATE] Dataset not found: {request.dataset_id}")
        raise HTTPException(status_code=404, detail="Dataset not found")

    print(f"[AUTO-ANNOTATE] Found dataset: {dataset.name}")

    # 데이터셋에 이미지가 있는지 확인
    images = db.query(Image).filter(Image.dataset_id == request.dataset_id).all()
    if not images:
        print(f"[AUTO-ANNOTATE] No images in dataset {request.dataset_id}")
        raise HTTPException(status_code=400, detail="No images in dataset")

    print(f"[AUTO-ANNOTATE] Found {len(images)} images")

    # 상태 업데이트
    dataset.status = DatasetStatus.PROCESSING
    dataset.auto_annotation_enabled = 1
    if request.model_id:
        dataset.auto_annotation_model_id = request.model_id
    db.commit()

    try:
        print("[AUTO-ANNOTATE] Loading auto-annotation service...")
        # 자동 어노테이션 서비스 로드
        service = get_auto_annotation_service()
        print("[AUTO-ANNOTATE] Service loaded successfully")

        # 모델 경로 결정
        model_path = None
        if request.model_id:
            print(f"[AUTO-ANNOTATE] Custom model requested: {request.model_id}")
            # 특정 모델 ID가 지정된 경우
            from app.models.model import Model
            model = db.query(Model).filter(Model.id == request.model_id).first()
            if model and model.file_path:
                model_path = model.file_path
                print(f"[AUTO-ANNOTATE] Using custom model: {model_path}")

        # 모델 로드 (model_path가 None이면 기본 경로 사용)
        print(f"[AUTO-ANNOTATE] Loading model from: {model_path or 'default (AI/models/best.pt)'}")
        if not service.load_model(model_path):
            print("[AUTO-ANNOTATE] Failed to load model!")
            raise HTTPException(
                status_code=500,
                detail="Failed to load model. Please ensure AI/models/best.pt exists or train the model first"
            )
        print("[AUTO-ANNOTATE] Model loaded successfully")

        # Confidence threshold (요청에서 받거나 기본값 0.25)
        conf_threshold = getattr(request, 'confidence_threshold', 0.25)
        print(f"[AUTO-ANNOTATE] Using confidence threshold: {conf_threshold}")

        # 각 이미지에 대해 자동 어노테이션 실행
        total_annotations = 0
        annotated_images_count = 0
        base_upload_dir = Path("uploads")
        print(f"[AUTO-ANNOTATE] Base upload directory: {base_upload_dir.absolute()}")

        for idx, img in enumerate(images, 1):
            img_path = base_upload_dir / img.file_path
            print(f"[AUTO-ANNOTATE] Processing image {idx}/{len(images)}: {img.filename}")
            print(f"[AUTO-ANNOTATE] Image path: {img_path}")

            if not img_path.exists():
                print(f"[AUTO-ANNOTATE] WARNING: Image file not found: {img_path}")
                continue

            # 추론 실행
            print(f"[AUTO-ANNOTATE] Running prediction...")
            annotations = service.predict_image(str(img_path), conf_threshold)
            print(f"[AUTO-ANNOTATE] Found {len(annotations)} annotations")

            # overwrite_existing이 True인 경우 기존 어노테이션 삭제
            if request.overwrite_existing:
                existing_annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
                if existing_annotations:
                    print(f"[AUTO-ANNOTATE] Overwrite mode: Deleting {len(existing_annotations)} existing annotations for image {img.id}")
                    for existing_ann in existing_annotations:
                        db.delete(existing_ann)
                    db.flush()  # Flush to ensure deletions are processed before inserts
                    print(f"[AUTO-ANNOTATE] Existing annotations deleted")

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
                print(f"[AUTO-ANNOTATE] Saved {len(annotations)} annotations for image {img.filename}")

        # 커밋
        print(f"[AUTO-ANNOTATE] Committing {total_annotations} annotations...")
        db.commit()
        print("[AUTO-ANNOTATE] Committed successfully")

        # 데이터셋 통계 업데이트
        print("[AUTO-ANNOTATE] Updating dataset statistics...")
        dataset.annotated_images = annotated_images_count
        dataset.status = DatasetStatus.ANNOTATED if annotated_images_count > 0 else DatasetStatus.READY

        # 실제로 detection된 클래스만 추출
        print("[AUTO-ANNOTATE] Extracting detected class names...")
        detected_annotations = db.query(Annotation).join(Image).filter(Image.dataset_id == request.dataset_id).all()
        detected_class_names = list(set([ann.class_name for ann in detected_annotations]))
        detected_class_names.sort()  # 알파벳 순 정렬

        print(f"[AUTO-ANNOTATE] Detected classes: {detected_class_names}")

        # 클래스 이름 업데이트 (실제로 detection된 클래스만)
        if detected_class_names:
            dataset.class_names = detected_class_names
            dataset.total_classes = len(detected_class_names)
            print(f"[AUTO-ANNOTATE] Updated class names to detected classes: {dataset.class_names}")
        else:
            print("[AUTO-ANNOTATE] No classes detected, keeping original class names")

        db.commit()

        # Invalidate datasets cache after auto-annotation
        invalidate_cache("datasets")
        print("[AUTO-ANNOTATE] Cache invalidated")

        result = {
            "message": "Auto-annotation completed",
            "dataset_id": request.dataset_id,
            "model_id": request.model_id,
            "status": "completed",
            "total_images": len(images),
            "annotated_images": annotated_images_count,
            "total_annotations": total_annotations,
            "confidence_threshold": conf_threshold
        }
        print(f"[AUTO-ANNOTATE] SUCCESS: {result}")
        return result

    except Exception as e:
        # 에러 발생 시 상태 복구
        print(f"[AUTO-ANNOTATE] ERROR: {str(e)}")
        print(f"[AUTO-ANNOTATE] Error type: {type(e).__name__}")
        import traceback
        print(f"[AUTO-ANNOTATE] Traceback:\n{traceback.format_exc()}")
        dataset.status = DatasetStatus.ERROR
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Auto-annotation failed: {str(e)}"
        )


@router.get("/{dataset_id}/images/{image_id}/download")
async def download_single_image(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Download a single image from a dataset
    
    Args:
        dataset_id: Dataset ID
        image_id: Image ID
        
    Returns:
        Image file as StreamingResponse
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get image record
    image = db.query(Image).filter(
        Image.id == image_id,
        Image.dataset_id == dataset_id
    ).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Download image from S3
        image_data = file_storage.download_from_s3(image.file_path)
        
        if not image_data:
            raise HTTPException(
                status_code=404,
                detail=f"Image file not found in storage: {image.filename}"
            )
        
        # Determine content type
        content_type = "image/jpeg"
        if image.filename.lower().endswith('.png'):
            content_type = "image/png"
        elif image.filename.lower().endswith('.gif'):
            content_type = "image/gif"
        elif image.filename.lower().endswith('.bmp'):
            content_type = "image/bmp"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{image.filename}"'
            }
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to download image {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download image: {str(e)}"
        )


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: int,
    include_annotations: bool = True,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Download entire dataset as a ZIP file
    
    Args:
        dataset_id: Dataset ID
        include_annotations: Whether to include annotation files (default: True)
        
    Returns:
        ZIP file containing all images and optionally annotations
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all images for this dataset
    images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
    
    if not images:
        raise HTTPException(
            status_code=404,
            detail="No images found in this dataset"
        )
    
    try:
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add dataset info
            dataset_info = f"""Dataset: {dataset.name}
Description: {dataset.description or 'N/A'}
Total Images: {len(images)}
Total Classes: {dataset.total_classes}
Class Names: {', '.join(dataset.class_names) if dataset.class_names else 'N/A'}
Created: {dataset.created_at}
"""
            zip_file.writestr("dataset_info.txt", dataset_info)
            
            # Add images
            failed_images = []
            for idx, image in enumerate(images):
                try:
                    # Download image from S3
                    image_data = file_storage.download_from_s3(image.file_path)
                    
                    if image_data:
                        # Add image to ZIP
                        zip_file.writestr(f"images/{image.filename}", image_data)
                        
                        # Add annotation if available and requested
                        if include_annotations and image.is_annotated:
                            annotations = db.query(Annotation).filter(
                                Annotation.image_id == image.id
                            ).all()
                            
                            if annotations:
                                # Create YOLO format annotation
                                annotation_lines = []
                                for ann in annotations:
                                    # YOLO format: class_id x_center y_center width height
                                    annotation_lines.append(
                                        f"{ann.class_id} {ann.x_center} {ann.y_center} "
                                        f"{ann.width} {ann.height}"
                                    )
                                
                                # Save annotation file (same name as image but .txt)
                                annotation_filename = image.filename.rsplit('.', 1)[0] + '.txt'
                                zip_file.writestr(
                                    f"annotations/{annotation_filename}",
                                    '\n'.join(annotation_lines)
                                )
                    else:
                        failed_images.append(image.filename)
                        print(f"[WARN] Failed to download image: {image.filename}")
                
                except Exception as e:
                    failed_images.append(image.filename)
                    print(f"[ERROR] Error processing image {image.filename}: {str(e)}")
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    print(f"[DOWNLOAD] Processed {idx + 1}/{len(images)} images")
            
            # Add failed images list if any
            if failed_images:
                zip_file.writestr(
                    "failed_images.txt",
                    "The following images could not be downloaded:\n" + 
                    '\n'.join(failed_images)
                )
        
        # Prepare response
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset.name}_{timestamp}.zip"
        
        print(f"[DOWNLOAD] Dataset {dataset_id} download complete: "
              f"{len(images) - len(failed_images)}/{len(images)} images")
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to create dataset download: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create dataset download: {str(e)}"
        )
