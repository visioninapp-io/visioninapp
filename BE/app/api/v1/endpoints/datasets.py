<<<<<<< HEAD
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form, Body
from fastapi.responses import StreamingResponse, Response, RedirectResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import io
import zipfile
import base64
from datetime import datetime
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.core.cache import cache_manager, invalidate_cache
from app.models.dataset import Dataset, Image, Annotation, DatasetStatus
from app.models.dataset_asset import DatasetAsset
from app.schemas.dataset import (
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetStats,
    ImageResponse, AnnotationCreate, AnnotationResponse, AutoAnnotationRequest
)
from app.schemas.dataset_asset import (
    DatasetAssetResponse, UploadCompleteItem as AssetUploadCompleteItem,
    UploadCompleteBatchRequest as AssetUploadCompleteBatchRequest
)
from app.utils.file_storage import file_storage
from app.services.dataset_cache_service import dataset_cache_service
from app.services.presigned_url_service import presigned_url_service
from app.core.config import settings
import time

router = APIRouter()


# ============================================================
# Pydantic Models for Presigned URL endpoints
# ============================================================

class UploadCompleteItemLegacy(BaseModel):
    """Legacy format for backward compatibility"""
    s3_key: str
    original_filename: str
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None

class UploadCompleteBatchRequestLegacy(BaseModel):
    """Legacy format for backward compatibility"""
    uploads: List[UploadCompleteItemLegacy]

class BatchUploadUrlRequest(BaseModel):
    filenames: List[str]
    content_type: Optional[str] = None

class UploadDatasetRequest(BaseModel):
    """Dataset upload with Presigned URL generation"""
    dataset_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    filenames: List[str]
    content_type: Optional[str] = None

class BatchDownloadUrlRequest(BaseModel):
    image_ids: List[int]


# ============================================================
# Dataset Endpoints
# ============================================================

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


@router.post("/presigned-upload-urls", status_code=status.HTTP_201_CREATED)
async def generate_presigned_upload_urls(
    request: UploadDatasetRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    데이터셋 생성 및 업로드용 Presigned URL 발급
    
    - dataset_id가 없으면: 새 데이터셋 생성 후 URL 발급
    - dataset_id가 있으면: 기존 데이터셋에 URL 발급
    - 클라이언트가 각 URL로 직접 S3에 병렬 업로드
    - URL 만료 시간: 30분
    
    사용 후:
    1. 받은 Presigned URL로 S3에 직접 PUT 요청 (파일 업로드)
    2. POST /api/v1/datasets/{dataset_id}/upload-complete-batch 호출 (업로드 완료 알림)
    """
    # Validate filenames
    if not request.filenames or len(request.filenames) == 0:
        raise HTTPException(status_code=400, detail="At least one filename is required")
    
    dataset = None
    
    # If dataset_id provided, use existing dataset
    if request.dataset_id:
        dataset = db.query(Dataset).filter(
            Dataset.id == request.dataset_id,
            Dataset.created_by == current_user["uid"]
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        # Create new dataset
        if not request.name:
            raise HTTPException(status_code=400, detail="Dataset name is required when creating new dataset")
        
        # Check if dataset name already exists for this user
        existing = db.query(Dataset).filter(
            Dataset.name == request.name,
            Dataset.created_by == current_user["uid"]
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        
        dataset = Dataset(
            name=request.name,
            description=request.description or "",
            dataset_type="object_detection",  # default
            total_images=0,
            annotated_images=0,
            total_classes=0,
            status=DatasetStatus.UPLOADING,
            created_by=current_user["uid"],
            auto_annotation_enabled=False,
            is_public=False
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
    
    try:
        # Generate presigned URLs
        urls = presigned_url_service.generate_batch_upload_urls(
            dataset_id=dataset.id,
            filenames=request.filenames,
            content_type=request.content_type,
            expiration=1800  # 30분
        )
        
        # Invalidate cache
        invalidate_cache("datasets")
        
        return {
            "success": True,
            "message": f"Presigned URLs generated for {len(request.filenames)} files",
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "status": dataset.status.value if hasattr(dataset.status, 'value') else dataset.status
            },
            "urls": urls,
            "total_count": len(urls)
        }
    except Exception as e:
        if dataset and not request.dataset_id:  # 새로 생성한 데이터셋이면 롤백
            db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URLs: {str(e)}"
        )


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
    """Trigger auto-annotation for a dataset using YOLO model (via AI service)"""
    print(f"[AUTO-ANNOTATE] Received request: dataset_id={request.dataset_id}, model_id={request.model_id}, conf={getattr(request, 'confidence_threshold', 0.25)}")

    from app.services.ai_client import get_ai_client
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
        print("[AUTO-ANNOTATE] Connecting to AI service...")
        ai_client = get_ai_client()
        
        # Check AI service health
        try:
            health = await ai_client.health_check()
            print(f"[AUTO-ANNOTATE] AI service available: {health.get('device_name')}")
        except Exception as e:
            print(f"[AUTO-ANNOTATE] AI service unavailable: {e}")
            raise HTTPException(status_code=503, detail="AI service unavailable")

        # 모델 경로 결정
        model_path = "AI/models/best.pt"  # Default fallback model
        
        if request.model_id:
            print(f"[AUTO-ANNOTATE] Custom model requested: model_id={request.model_id}")
            
            # First, try to find trained model in AI/uploads/models/
            trained_model_path = Path(f"AI/uploads/models/model_{request.model_id}/weights/best.pt")
            if trained_model_path.exists():
                model_path = str(trained_model_path)
                print(f"[AUTO-ANNOTATE] ✅ Using trained model: {model_path}")
            else:
                # Fallback: Try to get model from database
                print(f"[AUTO-ANNOTATE] Trained model not found, checking database...")
                from app.models.model import Model
                model = db.query(Model).filter(Model.id == request.model_id).first()
                if model and model.file_path:
                    model_path = model.file_path
                    print(f"[AUTO-ANNOTATE] ✅ Using DB model path: {model_path}")
                else:
                    print(f"[AUTO-ANNOTATE] ⚠️ Model {request.model_id} not found, using default model")
        else:
            print(f"[AUTO-ANNOTATE] No model_id specified, using default model")

        print(f"[AUTO-ANNOTATE] Final model path: {model_path}")

        # Confidence threshold
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

            # 추론 실행 (AI service를 통해)
            print(f"[AUTO-ANNOTATE] Running prediction via AI service...")
            try:
                result = await ai_client.predict(
                    model_path=model_path,
                    image_path=str(img_path),
                    conf=conf_threshold
                )
                annotations = result.get('predictions', [])
                print(f"[AUTO-ANNOTATE] Found {len(annotations)} annotations")
            except Exception as e:
                print(f"[AUTO-ANNOTATE] Prediction failed: {e}")
                continue

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
                    # AI service returns Detection objects, convert to dict if needed
                    if isinstance(ann, dict):
                        bbox = ann['bbox']
                        class_id = ann['class_id']
                        class_name = ann['class_name']
                        confidence = ann['confidence']
                    else:
                        # Pydantic model from AI service
                        bbox = ann.bbox
                        class_id = ann.class_id
                        class_name = ann.class_name
                        confidence = ann.confidence
                    
                    db_annotation = Annotation(
                        image_id=img.id,
                        class_id=class_id,
                        class_name=class_name,
                        x_center=bbox['x_center'] if isinstance(bbox, dict) else bbox.get('x_center'),
                        y_center=bbox['y_center'] if isinstance(bbox, dict) else bbox.get('y_center'),
                        width=bbox['width'] if isinstance(bbox, dict) else bbox.get('width'),
                        height=bbox['height'] if isinstance(bbox, dict) else bbox.get('height'),
                        confidence=confidence,
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
    Download entire dataset as a ZIP file (images + videos)
    
    Args:
        dataset_id: Dataset ID
        include_annotations: Whether to include annotation files (default: True)
        
    Returns:
        ZIP file containing all images, videos and optionally annotations
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all assets (images + videos) from DatasetAsset table
    assets = db.query(DatasetAsset).filter(
        DatasetAsset.dataset_id == dataset_id
    ).all()
    
    # Also get images from Image table for backward compatibility
    images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
    
    if not assets and not images:
        raise HTTPException(
            status_code=404,
            detail="No files found in this dataset"
        )
    
    try:
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add dataset info
            total_images = len([a for a in assets if a.kind == 'image']) + len(images)
            total_videos = len([a for a in assets if a.kind == 'video'])
            
            dataset_info = f"""Dataset: {dataset.name}
Description: {dataset.description or 'N/A'}
Total Images: {total_images}
Total Videos: {total_videos}
Total Classes: {dataset.total_classes}
Class Names: {', '.join(dataset.class_names) if dataset.class_names else 'N/A'}
Created: {dataset.created_at}
"""
            zip_file.writestr("dataset_info.txt", dataset_info)
            
            # Add assets (images + videos from DatasetAsset)
            failed_files = []
            processed_count = 0
            
            for idx, asset in enumerate(assets):
                try:
                    # Download file from S3
                    file_data = file_storage.download_from_s3(asset.s3_key)
                    
                    if file_data:
                        # Add file to ZIP in appropriate folder
                        folder = "images" if asset.kind == 'image' else "videos"
                        zip_file.writestr(f"{folder}/{asset.original_filename}", file_data)
                        processed_count += 1
                        
                        # Add annotation if available and requested (images only)
                        if include_annotations and asset.kind == 'image':
                            # Try to find corresponding Image record for annotations
                            image_record = None
                            for img in images:
                                if img.file_path == asset.s3_key or img.filename == asset.original_filename:
                                    image_record = img
                                    break
                            
                            if image_record and image_record.is_annotated:
                                annotations = db.query(Annotation).filter(
                                    Annotation.image_id == image_record.id
                                ).all()
                                
                                if annotations:
                                    # Create YOLO format annotation
                                    annotation_lines = []
                                    for ann in annotations:
                                        annotation_lines.append(
                                            f"{ann.class_id} {ann.x_center} {ann.y_center} "
                                            f"{ann.width} {ann.height}"
                                        )
                                    
                                    annotation_filename = asset.original_filename.rsplit('.', 1)[0] + '.txt'
                                    zip_file.writestr(
                                        f"annotations/{annotation_filename}",
                                        '\n'.join(annotation_lines)
                                    )
                    else:
                        failed_files.append(asset.original_filename)
                        print(f"[WARN] Failed to download asset: {asset.original_filename}")
                
                except Exception as e:
                    failed_files.append(asset.original_filename)
                    print(f"[ERROR] Error processing asset {asset.original_filename}: {str(e)}")
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    print(f"[DOWNLOAD] Processed {idx + 1}/{len(assets)} assets")
            
            # Also add images from Image table (for backward compatibility with old data)
            for idx, image in enumerate(images):
                # Skip if already added from DatasetAsset
                asset_exists = any(asset.s3_key == image.file_path for asset in assets)
                if asset_exists:
                    continue
                
                try:
                    # Download image from S3
                    image_data = file_storage.download_from_s3(image.file_path)
                    
                    if image_data:
                        # Add image to ZIP
                        zip_file.writestr(f"images/{image.filename}", image_data)
                        processed_count += 1
                        
                        # Add annotation if available and requested
                        if include_annotations and image.is_annotated:
                            annotations = db.query(Annotation).filter(
                                Annotation.image_id == image.id
                            ).all()
                            
                            if annotations:
                                annotation_lines = []
                                for ann in annotations:
                                    annotation_lines.append(
                                        f"{ann.class_id} {ann.x_center} {ann.y_center} "
                                        f"{ann.width} {ann.height}"
                                    )
                                
                                annotation_filename = image.filename.rsplit('.', 1)[0] + '.txt'
                                zip_file.writestr(
                                    f"annotations/{annotation_filename}",
                                    '\n'.join(annotation_lines)
                                )
                    else:
                        failed_files.append(image.filename)
                        print(f"[WARN] Failed to download image: {image.filename}")
                
                except Exception as e:
                    failed_files.append(image.filename)
                    print(f"[ERROR] Error processing image {image.filename}: {str(e)}")
            
            # Add failed files list if any
            if failed_files:
                zip_file.writestr(
                    "failed_files.txt",
                    "The following files could not be downloaded:\n" + 
                    '\n'.join(failed_files)
                )
        
        # Prepare response
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset.name}_{timestamp}.zip"
        
        print(f"[DOWNLOAD] Dataset {dataset_id} download complete: "
              f"{processed_count} files processed, {len(failed_files)} failed")
        
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


# ============================================================
# Presigned URL 기반 업로드/다운로드 엔드포인트
# ============================================================

@router.post("/{dataset_id}/upload-complete-batch")
async def confirm_upload_complete_batch(
    dataset_id: int,
    request: AssetUploadCompleteBatchRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    S3 업로드 완료 알림 - DatasetAsset 테이블에 메타데이터 저장
    
    items: [
        {
            "s3_key": "datasets/dataset_1/images/xxx.jpg",
            "original_filename": "photo.jpg",
            "file_size": 123456,
            "width": 1920,
            "height": 1080,
            "duration_ms": null  # for videos
        },
        ...
    ]
    
    - 이미지와 동영상을 자동으로 구분하여 DatasetAsset에 저장
    - kind는 s3_key의 경로(images/ or videos/)로 자동 판별
    """
    # 데이터셋 존재 및 권한 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.created_by == current_user["uid"]
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    successful = []
    failed = []
    
    try:
        for item in request.items:
            try:
                s3_key = item.s3_key
                
                # S3 파일 존재 확인
                if not presigned_url_service.check_object_exists(s3_key):
                    failed.append({
                        "filename": item.original_filename,
                        "error": "File not found in S3"
                    })
                    continue
                
                # kind 자동 판별 (s3_key 경로에서 추출)
                if '/videos/' in s3_key:
                    kind = 'video'
                elif '/images/' in s3_key:
                    kind = 'image'
                else:
                    kind = 'image'  # default
                
                # DatasetAsset에 메타데이터 저장
                asset = DatasetAsset(
                    dataset_id=dataset_id,
                    kind=kind,
                    original_filename=item.original_filename,
                    s3_key=s3_key,
                    file_size=item.file_size,
                    width=item.width,
                    height=item.height,
                    duration_ms=item.duration_ms,
                    created_by=current_user["uid"]
                )
                
                db.add(asset)
                
                # 하위 호환성: Image 테이블에도 저장 (이미지만)
                if kind == 'image':
                    image = Image(
                        dataset_id=dataset_id,
                        filename=item.original_filename,
                        file_path=s3_key,
                        file_size=item.file_size,
                        width=item.width,
                        height=item.height,
                        is_annotated=False
                    )
                    db.add(image)
                
                successful.append({
                    "filename": item.original_filename,
                    "s3_key": s3_key,
                    "kind": kind
                })
                
            except Exception as e:
                failed.append({
                    "filename": getattr(item, "original_filename", "unknown"),
                    "error": str(e)
                })
        
        # 데이터셋 통계 업데이트
        image_count = db.query(DatasetAsset).filter(
            DatasetAsset.dataset_id == dataset_id,
            DatasetAsset.kind == 'image'
        ).count()
        
        dataset.total_images = image_count
        
        db.commit()
        
        # 캐시 무효화
        invalidate_cache(f"dataset_{dataset_id}")
        dataset_cache_service.invalidate_dataset_cache(dataset_id)
        
        return {
            "success": True,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful": successful,
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process batch upload: {str(e)}"
        )


@router.get("/{dataset_id}/assets", response_model=List[DatasetAssetResponse])
async def get_dataset_assets(
    dataset_id: int,
    kind: Optional[str] = None,  # "image" | "video" | None (all)
    page: int = 1,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    데이터셋의 assets(이미지/동영상) 목록 조회
    
    - kind: "image", "video", 또는 None(전체)
    - 페이징 지원
    """
    # 데이터셋 권한 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.created_by == current_user["uid"]
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 쿼리 구성
    query = db.query(DatasetAsset).filter(DatasetAsset.dataset_id == dataset_id)
    
    if kind:
        query = query.filter(DatasetAsset.kind == kind)
    
    # 페이징
    offset = (page - 1) * limit
    assets = query.order_by(DatasetAsset.created_at.desc()).offset(offset).limit(limit).all()
    
    return assets


@router.get("/assets/{asset_id}/presigned-download")
async def get_asset_download_url(
    asset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    DatasetAsset 단건 다운로드용 Presigned URL 발급
    
    - 이미지/동영상 모두 지원
    - URL 만료 시간: 1시간
    """
    # Asset 조회
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == asset_id).first()
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # 데이터셋 권한 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == asset.dataset_id,
        Dataset.created_by == current_user["uid"]
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Presigned GET URL 생성
        url_data = presigned_url_service.generate_download_url(
            s3_key=asset.s3_key,
            expiration=3600,
            filename=asset.original_filename
        )
        
        return {
            "success": True,
            "asset_id": asset.id,
            "original_filename": asset.original_filename,
            "kind": asset.kind,
            "download_url": url_data["download_url"],
            "expires_in": url_data["expires_in"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )


@router.post("/{dataset_id}/presigned-download-urls-batch")
async def generate_presigned_download_urls_batch(
    dataset_id: int,
    request: BatchDownloadUrlRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    이미지 다운로드용 Presigned URL 발급 (단일/배치 모두 처리)
    
    - 단일 파일: image_ids에 1개만 전달
    - 여러 파일: image_ids에 여러 개 전달
    - 클라이언트가 각 URL로 직접 S3에서 다운로드
    - URL 만료 시간: 1시간
    """
    # 데이터셋 권한 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.created_by == current_user["uid"]
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 이미지 조회
    images = db.query(Image).filter(
        Image.id.in_(request.image_ids),
        Image.dataset_id == dataset_id
    ).all()
    
    if not images:
        raise HTTPException(status_code=404, detail="No images found")
    
    # 이미지 ID로 매핑
    image_map = {img.id: img for img in images}
    
    # Presigned URL 생성
    urls = []
    failed = []
    
    for image_id in request.image_ids:
        image = image_map.get(image_id)
        
        if not image:
            failed.append({
                "image_id": image_id,
                "error": "Image not found"
            })
            continue
        
        try:
            url_data = presigned_url_service.generate_download_url(
                s3_key=image.file_path,
                filename=image.filename,
                expiration=3600  # 1시간
            )
            
            urls.append({
                "image_id": image.id,
                "download_url": url_data["download_url"],
                "filename": image.filename,
                "s3_key": image.file_path,
                "expires_in": url_data["expires_in"],
                "generation_time": url_data["generation_time"]
            })
        except Exception as e:
            failed.append({
                "image_id": image_id,
                "error": str(e)
            })
    
    return {
        "success": True,
        "urls": urls,
        "total_count": len(urls),
        "failed_count": len(failed),
        "failed": failed
    }
=======
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
    db.delete(dataset)
    db.commit()

    # Invalidate cache to ensure frontend gets updated list
    invalidate_cache("datasets")
    print("[CACHE] Invalidated after dataset deletion: datasets")

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

    # Convert is_annotated from int to bool for response
    for img in images:
        img.is_annotated = bool(img.is_annotated)

    print(f"[GET IMAGES] Returning {len(images)} images for dataset {dataset_id}")
    for img in images:
        print(f"  - {img.filename}: is_annotated={img.is_annotated}")

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
        print("[CACHE] Invalidated after upload: datasets")

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
>>>>>>> feature/llm-pipeline
