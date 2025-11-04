from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form, Body
from fastapi.responses import StreamingResponse, Response, RedirectResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import io
import zipfile
import base64
from datetime import datetime
import hashlib
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.dataset import Dataset, Annotation, DatasetVersion
from app.models.label_class import LabelClass
from app.utils.project_helper import get_or_create_default_project
from app.utils.dataset_helper import (
    get_or_create_dataset_version,
    get_or_create_dataset_split,
    get_latest_dataset_version,
    get_assets_from_dataset,
    format_assets_as_images,
    format_asset_as_image
)
from app.models.asset import Asset, AssetType
from app.models.dataset_split import DatasetSplit, DatasetSplitType
from app.schemas.dataset import (
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetStats,
    AnnotationCreate, AnnotationResponse, AutoAnnotationRequest
)
from app.schemas.dataset_asset import (
    DatasetAssetResponse, UploadCompleteItem as AssetUploadCompleteItem,
    UploadCompleteBatchRequest as AssetUploadCompleteBatchRequest
)
from app.utils.file_storage import file_storage
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
    asset_ids: List[int]


# ============================================================
# Dataset Endpoints
# ============================================================

@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get overall dataset statistics for current user"""
    # Get user's default project
    from app.models.project import Project
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    # Get datasets from user's project
    user_datasets = db.query(Dataset).filter(Dataset.project_id == default_project.id).all()
    dataset_ids = [d.id for d in user_datasets]
    
    # Asset에서 통계 계산
    total_assets = 0
    total_images = 0
    
    if dataset_ids:
        for dataset_id in dataset_ids:
            # 전체 Asset 개수 (이미지 + 비디오)
            all_assets, _ = get_assets_from_dataset(
                db=db,
                dataset_id=dataset_id,
                asset_type=None,  # 전체
                limit=100000
            )
            total_assets += len(all_assets)
            
            # 이미지 개수
            image_assets, _ = get_assets_from_dataset(
                db=db,
                dataset_id=dataset_id,
                asset_type=AssetType.IMAGE,
                limit=100000
            )
            total_images += len(image_assets)
    
    # 전체 Annotation 개수 계산
    total_annotations = db.query(Annotation).join(Asset).join(DatasetSplit).join(DatasetVersion).filter(
        DatasetVersion.dataset_id.in_(dataset_ids)
    ).count() if dataset_ids else 0
    
    total_datasets = len(user_datasets)

    return {
        "total_assets": total_assets,
        "total_images": total_images,
        "total_datasets": total_datasets,
        "total_annotations": total_annotations
    }


@router.get("/", response_model=List[DatasetResponse])
async def get_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all datasets for current user"""
    # Get user's default project
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    # Query datasets from user's project
    datasets = db.query(Dataset).filter(
        Dataset.project_id == default_project.id
    ).offset(skip).limit(limit).all()

    # Convert to response format with computed fields
    datasets_list = []
    for d in datasets:
        # Asset 개수 계산
        assets, _ = get_assets_from_dataset(
            db=db,
            dataset_id=d.id,
            asset_type=None,  # 전체
            limit=100000
        )
        total_assets = len(assets)
        
        # Annotation 개수 계산
        total_annotations = db.query(Annotation).join(Asset).join(DatasetSplit).join(DatasetVersion).filter(
            DatasetVersion.dataset_id == d.id
        ).count()
        
        # 버전 개수
        version_count = len(d.versions)
        
        datasets_list.append({
            "id": d.id,
            "project_id": d.project_id,
            "name": d.name,
            "description": d.description,
            "created_at": d.created_at,
            "total_assets": total_assets,
            "total_annotations": total_annotations,
            "version_count": version_count
        })

    return datasets_list


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new dataset (ERD 기준)"""
    # Get or create default project for user
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    # Check if dataset name already exists in this project
    db_dataset = db.query(Dataset).filter(
        Dataset.name == dataset.name,
        Dataset.project_id == default_project.id
    ).first()
    if db_dataset:
        raise HTTPException(status_code=400, detail="Dataset with this name already exists in this project")

    # Create dataset with ERD fields only
    db_dataset = Dataset(
        project_id=default_project.id,
        name=dataset.name,
        description=dataset.description
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    # Return with computed fields
    return {
        "id": db_dataset.id,
        "project_id": db_dataset.project_id,
        "name": db_dataset.name,
        "description": db_dataset.description,
        "created_at": db_dataset.created_at,
        "total_assets": 0,
        "total_annotations": 0,
        "version_count": 0
    }


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get a specific dataset with computed fields"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Asset 개수 계산
    assets, _ = get_assets_from_dataset(
        db=db,
        dataset_id=dataset.id,
        asset_type=None,
        limit=100000
    )
    total_assets = len(assets)
    
    # Annotation 개수 계산
    total_annotations = db.query(Annotation).join(Asset).join(DatasetSplit).join(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset.id
    ).count()
    
    # 버전 개수
    version_count = len(dataset.versions)
    
    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "name": dataset.name,
        "description": dataset.description,
        "created_at": dataset.created_at,
        "total_assets": total_assets,
        "total_annotations": total_annotations,
        "version_count": version_count
    }


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update a dataset (ERD 기준)"""
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Update only allowed fields
    update_data = dataset_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(db_dataset, field):
            setattr(db_dataset, field, value)

    db.commit()
    db.refresh(db_dataset)
    
    # Return with computed fields
    assets, _ = get_assets_from_dataset(db=db, dataset_id=db_dataset.id, asset_type=None, limit=100000)
    total_assets = len(assets)
    total_annotations = db.query(Annotation).join(Asset).join(DatasetSplit).join(DatasetVersion).filter(
        DatasetVersion.dataset_id == db_dataset.id
    ).count()
    version_count = len(db_dataset.versions)
    
    return {
        "id": db_dataset.id,
        "project_id": db_dataset.project_id,
        "name": db_dataset.name,
        "description": db_dataset.description,
        "created_at": db_dataset.created_at,
        "total_assets": total_assets,
        "total_annotations": total_annotations,
        "version_count": version_count
    }


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
    Get images in a dataset with base64 encoding
    
    Args:
        dataset_id: Dataset ID
        page: Page number (1-based, default: 1)
        limit: Number of images per page (default: 50)
    
    Returns:
        Dictionary with base64 encoded images and metadata
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Asset에서 조회
    assets, total_count = get_assets_from_dataset(
        db=db,
        dataset_id=dataset_id,
        asset_type=AssetType.IMAGE,
        page=page,
        limit=limit
    )
    
    if not assets:
        return {
            "dataset_id": dataset_id,
            "page": page,
            "page_size": limit,
            "cached": False,
            "images": [],
            "total_images": 0,
            "total_pages": 0
        }
    
    print(f"[GET IMAGES] Dataset {dataset_id}: {len(assets)} assets, page {page}, limit {limit}")
    
    # Asset을 Image 형식으로 변환
    image_list = format_assets_as_images(assets)
    
    return {
        "dataset_id": dataset_id,
        "page": page,
        "page_size": limit,
        "cached": False,
        "images": image_list,
        "total_images": total_count,
        "total_pages": (total_count + limit - 1) // limit
    }


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
    
    # Get or create default project for user
    default_project = get_or_create_default_project(db, current_user["uid"])
    
    # If dataset_id provided, use existing dataset
    if request.dataset_id:
        dataset = db.query(Dataset).filter(
            Dataset.id == request.dataset_id,
            Dataset.project_id == default_project.id
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        # Create new dataset
        if not request.name:
            raise HTTPException(status_code=400, detail="Dataset name is required when creating new dataset")
        
        # Check if dataset name already exists in this project
        existing = db.query(Dataset).filter(
            Dataset.name == request.name,
            Dataset.project_id == default_project.id
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        
        # Create dataset with ERD fields only
        dataset = Dataset(
            name=request.name,
            description=request.description or "",
            project_id=default_project.id
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
        
        return {
            "success": True,
            "message": f"Presigned URLs generated for {len(request.filenames)} files",
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description
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
        # Upload images to S3
        successful_uploads, failed_uploads = await file_storage.save_images_batch(files, dataset.id)

        # DatasetVersion 및 Split 생성/조회
        dataset_version = get_or_create_dataset_version(db, dataset.id)
        dataset_split = get_or_create_dataset_split(
            db,
            dataset_version.id,
            DatasetSplitType.UNASSIGNED
        )

        # Asset 생성
        for upload in successful_uploads:
            metadata = upload["metadata"]
            storage_uri = metadata["relative_path"]
            
            # SHA256 계산
            sha256_hash = calculate_sha256_from_s3(storage_uri)
            
            asset = Asset(
                dataset_split_id=dataset_split.id,
                type=AssetType.IMAGE,
                storage_uri=storage_uri,
                sha256=sha256_hash,
                bytes=metadata["file_size"],
                width=metadata.get("width"),
                height=metadata.get("height"),
                duration_ms=None,
                fps=None,
                frame=None,
                codec=None
            )
            db.add(asset)

        db.commit()

        # 전체 Asset 개수 계산
        total_assets = db.query(Asset).join(DatasetSplit).filter(
            DatasetSplit.dataset_version_id == dataset_version.id
        ).count()

        response = {
            "message": f"Successfully uploaded {len(successful_uploads)} images",
            "dataset_id": dataset_id,
            "total_assets": total_assets,
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads)
        }

        if failed_uploads:
            response["failed_files"] = failed_uploads

        return response

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/images/{image_id}/annotations", response_model=List[AnnotationResponse])
async def get_image_annotations(
    image_id: int,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Get all annotations for a specific asset
    """
    # Asset 조회
    asset = db.query(Asset).filter(Asset.id == image_id).first()
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # Asset 기반 Annotation 조회
    query = db.query(Annotation).filter(Annotation.asset_id == asset.id)

    # Apply confidence threshold filter if specified
    if min_confidence is not None:
        query = query.filter(Annotation.confidence >= min_confidence)
        print(f"[GET ANNOTATIONS] Filtering by confidence >= {min_confidence}")

    annotations = query.all()
    print(f"[GET ANNOTATIONS] Found {len(annotations)} annotations for asset {image_id}")

    return annotations


@router.post("/annotations", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
async def create_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Create a new annotation
    """
    # Asset 존재 확인
    asset = db.query(Asset).filter(Asset.id == annotation.asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # LabelClass 존재 확인
    label_class = db.query(LabelClass).filter(LabelClass.id == annotation.label_class_id).first()
    if not label_class:
        raise HTTPException(status_code=404, detail="Label class not found")
    
    # Annotation 생성
    db_annotation = Annotation(
        asset_id=annotation.asset_id,
        label_class_id=annotation.label_class_id,
        model_version_id=annotation.model_version_id,
        geometry_type=annotation.geometry_type,
        geometry=annotation.geometry,
        is_normalized=annotation.is_normalized,
        source=annotation.source,
        confidence=annotation.confidence,
        annotator_name=annotation.annotator_name or current_user.get("name", "system")
    )
    db.add(db_annotation)
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

    # Asset에서 이미지 조회
    assets, _ = get_assets_from_dataset(
        db=db,
        dataset_id=request.dataset_id,
        asset_type=AssetType.IMAGE,
        limit=100000
    )
    
    if not assets:
        print(f"[AUTO-ANNOTATE] No images in dataset {request.dataset_id}")
        raise HTTPException(status_code=400, detail="No images in dataset")

    print(f"[AUTO-ANNOTATE] Found {len(assets)} assets")

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

        # 각 Asset에 대해 자동 어노테이션 실행
        total_annotations = 0
        annotated_images_count = 0
        base_upload_dir = Path("uploads")
        print(f"[AUTO-ANNOTATE] Base upload directory: {base_upload_dir.absolute()}")

        # DatasetVersion에서 LabelClass 찾기
        dataset_version = get_latest_dataset_version(db, request.dataset_id)
        if not dataset_version:
            raise HTTPException(status_code=400, detail="Dataset version not found")
        
        ontology = dataset_version.ontology_version
        label_classes = db.query(LabelClass).filter(
            LabelClass.ontology_version_id == ontology.id
        ).all()
        label_class_map = {lc.display_name: lc for lc in label_classes}

        for idx, asset in enumerate(assets, 1):
            # S3 경로 또는 로컬 경로 처리
            if asset.storage_uri.startswith('datasets/'):
                img_path = base_upload_dir / asset.storage_uri
            else:
                img_path = Path(asset.storage_uri)
            
            filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
            print(f"[AUTO-ANNOTATE] Processing asset {idx}/{len(assets)}: {filename}")
            print(f"[AUTO-ANNOTATE] Asset path: {img_path}")

            if not img_path.exists():
                print(f"[AUTO-ANNOTATE] WARNING: Asset file not found: {img_path}")
                continue

            # 추론 실행 (AI service를 통해)
            print(f"[AUTO-ANNOTATE] Running prediction via AI service...")
            try:
                result = await ai_client.predict(
                    model_path=model_path,
                    image_path=str(img_path),
                    conf=conf_threshold
                )
                predictions = result.get('predictions', [])
                print(f"[AUTO-ANNOTATE] Found {len(predictions)} predictions")
            except Exception as e:
                print(f"[AUTO-ANNOTATE] Prediction failed: {e}")
                continue

            # overwrite_existing이 True인 경우 기존 어노테이션 삭제
            if request.overwrite_existing:
                existing_annotations = db.query(Annotation).filter(Annotation.asset_id == asset.id).all()
                if existing_annotations:
                    print(f"[AUTO-ANNOTATE] Overwrite mode: Deleting {len(existing_annotations)} existing annotations for asset {asset.id}")
                    for existing_ann in existing_annotations:
                        db.delete(existing_ann)
                    db.flush()
                    print(f"[AUTO-ANNOTATE] Existing annotations deleted")

            if predictions:
                # 어노테이션 저장
                for pred in predictions:
                    # AI service returns Detection objects, convert to dict if needed
                    if isinstance(pred, dict):
                        bbox = pred['bbox']
                        class_id = pred['class_id']
                        class_name = pred['class_name']
                        confidence = pred['confidence']
                    else:
                        # Pydantic model from AI service
                        bbox = pred.bbox
                        class_id = pred.class_id
                        class_name = pred.class_name
                        confidence = pred.confidence
                    
                    # LabelClass 찾기
                    label_class = label_class_map.get(class_name)
                    if not label_class:
                        print(f"[AUTO-ANNOTATE] Warning: Label class '{class_name}' not found in ontology, skipping")
                        continue
                    
                    # Geometry 데이터 구성
                    geometry_data = {
                        "bbox": {
                            "x_center": bbox['x_center'] if isinstance(bbox, dict) else bbox.get('x_center', 0),
                            "y_center": bbox['y_center'] if isinstance(bbox, dict) else bbox.get('y_center', 0),
                            "width": bbox['width'] if isinstance(bbox, dict) else bbox.get('width', 0),
                            "height": bbox['height'] if isinstance(bbox, dict) else bbox.get('height', 0)
                        }
                    }
                    
                    from app.models.dataset import GeometryType
                    
                    db_annotation = Annotation(
                        asset_id=asset.id,
                        label_class_id=label_class.id,
                        geometry_type=GeometryType.BBOX,
                        geometry=geometry_data,
                        is_normalized=True,
                        source="model",
                        confidence=confidence,
                        annotator_name="auto-annotation"
                    )
                    db.add(db_annotation)
                    total_annotations += 1

                annotated_images_count += 1
                print(f"[AUTO-ANNOTATE] Saved {len(predictions)} annotations for asset {filename}")

        # 커밋
        print(f"[AUTO-ANNOTATE] Committing {total_annotations} annotations...")
        db.commit()
        print("[AUTO-ANNOTATE] Committed successfully")

        # Commit all annotations
        db.commit()
        
        print(f"[AUTO-ANNOTATE] Successfully created {total_annotations} annotations for {annotated_images_count} images")

        result = {
            "message": "Auto-annotation completed",
            "dataset_id": request.dataset_id,
            "model_id": request.model_id,
            "status": "completed",
            "total_images": len(assets),
            "annotated_images": annotated_images_count,
            "total_annotations": total_annotations,
            "confidence_threshold": conf_threshold
        }
        print(f"[AUTO-ANNOTATE] SUCCESS: {result}")
        return result

    except Exception as e:
        # 에러 발생 시 롤백
        print(f"[AUTO-ANNOTATE] ERROR: {str(e)}")
        print(f"[AUTO-ANNOTATE] Error type: {type(e).__name__}")
        import traceback
        print(f"[AUTO-ANNOTATE] Traceback:\n{traceback.format_exc()}")
        db.rollback()

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
    Download a single asset from a dataset
    
    Args:
        dataset_id: Dataset ID
        image_id: Asset ID
        
    Returns:
        Asset file as StreamingResponse
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Asset 조회
    asset = db.query(Asset).join(DatasetSplit).join(DatasetVersion).filter(
            Asset.id == image_id,
        DatasetVersion.dataset_id == dataset_id
        ).first()
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    try:
        # Download asset from S3
        asset_data = file_storage.download_from_s3(asset.storage_uri)
        
        if not asset_data:
            raise HTTPException(
                status_code=404,
                detail=f"Asset file not found in storage: {asset.storage_uri}"
            )
        
        # Determine content type
        filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
        content_type = "image/jpeg"
        if filename.lower().endswith('.png'):
            content_type = "image/png"
        elif filename.lower().endswith('.gif'):
            content_type = "image/gif"
        elif filename.lower().endswith('.bmp'):
            content_type = "image/bmp"
        elif filename.lower().endswith('.mp4'):
            content_type = "video/mp4"
        elif filename.lower().endswith('.avi'):
            content_type = "video/avi"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(asset_data),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to download asset {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download asset: {str(e)}"
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
    
    # Asset 테이블에서 조회
    assets, total_count = get_assets_from_dataset(
        db=db,
        dataset_id=dataset_id,
        limit=10000  # 큰 수로 설정하여 모든 asset 가져오기
    )
    
    if not assets:
        raise HTTPException(
            status_code=404,
            detail="No files found in this dataset"
        )
    
    try:
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add dataset info
            total_images = len([a for a in assets if a.type == AssetType.IMAGE])
            total_videos = len([a for a in assets if a.type == AssetType.VIDEO])
            
            # Get class names from latest dataset version
            from app.models.label_class import LabelClass
            
            latest_version = db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset.id
            ).order_by(DatasetVersion.created_at.desc()).first()
            
            class_names = []
            if latest_version and latest_version.ontology_version:
                label_classes = db.query(LabelClass).filter(
                    LabelClass.ontology_version_id == latest_version.ontology_version_id
                ).all()
                class_names = [lc.display_name for lc in label_classes]
            
            dataset_info = f"""Dataset: {dataset.name}
Description: {dataset.description or 'N/A'}
Total Images: {total_images}
Total Videos: {total_videos}
Total Classes: {len(class_names)}
Class Names: {', '.join(class_names) if class_names else 'N/A'}
Created: {dataset.created_at}
"""
            zip_file.writestr("dataset_info.txt", dataset_info)
            
            # Asset 처리
            failed_files = []
            processed_count = 0
            
            for idx, asset in enumerate(assets):
                try:
                    # Download file from S3
                    file_data = file_storage.download_from_s3(asset.storage_uri)
                    
                    if file_data:
                        # Add file to ZIP in appropriate folder
                        folder = "images" if asset.type == AssetType.IMAGE else "videos"
                        filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
                        zip_file.writestr(f"{folder}/{filename}", file_data)
                        processed_count += 1
                        
                        # Add annotation if available and requested (images only)
                        if include_annotations and asset.type == AssetType.IMAGE:
                            annotations = db.query(Annotation).filter(
                                Annotation.asset_id == asset.id
                            ).all()
                            
                            if annotations:
                                # Create YOLO format annotation
                                annotation_lines = []
                                for ann in annotations:
                                    # LabelClass에서 class_id 가져오기
                                    label_class = db.query(LabelClass).filter(LabelClass.id == ann.label_class_id).first()
                                    class_id = label_class.id if label_class else 0
                                    
                                    # Geometry에서 좌표 추출
                                    if ann.geometry and isinstance(ann.geometry, dict):
                                        bbox = ann.geometry.get('bbox', {})
                                        x_center = bbox.get('x_center', 0)
                                        y_center = bbox.get('y_center', 0)
                                        width = bbox.get('width', 0)
                                        height = bbox.get('height', 0)
                                    else:
                                        # Geometry가 없으면 스킵
                                        continue
                                    
                                    annotation_lines.append(
                                        f"{class_id} {x_center} {y_center} {width} {height}"
                                    )
                                
                                annotation_filename = filename.rsplit('.', 1)[0] + '.txt'
                                zip_file.writestr(
                                    f"annotations/{annotation_filename}",
                                    '\n'.join(annotation_lines)
                                )
                    else:
                        failed_files.append(filename)
                        print(f"[WARN] Failed to download asset: {filename}")
                
                except Exception as e:
                    failed_files.append(asset.storage_uri)
                    print(f"[ERROR] Error processing asset {asset.storage_uri}: {str(e)}")
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    print(f"[DOWNLOAD] Processed {idx + 1}/{len(assets)} assets")
            
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

def calculate_sha256_from_s3(s3_key: str) -> str:
    """S3 파일의 SHA256 해시 계산"""
    try:
        # S3에서 파일 다운로드하여 해시 계산
        file_data = file_storage.download_from_s3(s3_key)
        if file_data:
            return hashlib.sha256(file_data).hexdigest()
        else:
            # 파일이 없으면 s3_key 기반 임시 해시 생성
            return hashlib.sha256(s3_key.encode()).hexdigest()[:64]
    except Exception as e:
        print(f"[WARN] Failed to calculate SHA256 for {s3_key}: {e}")
        # 실패 시 s3_key 기반 임시 해시
        return hashlib.sha256(s3_key.encode()).hexdigest()[:64]


@router.post("/{dataset_id}/upload-complete-batch")
async def confirm_upload_complete_batch(
    dataset_id: int,
    request: AssetUploadCompleteBatchRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    S3 업로드 완료 알림 - Asset 테이블에 메타데이터 저장
    
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
    """
    # 데이터셋 존재 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # DatasetVersion 및 Split 생성/조회
    dataset_version = get_or_create_dataset_version(db, dataset_id)
    dataset_split = get_or_create_dataset_split(
        db, 
        dataset_version.id, 
        DatasetSplitType.UNASSIGNED
    )
    
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
                
                # Asset 타입 판별
                if '/videos/' in s3_key:
                    asset_type = AssetType.VIDEO
                elif '/images/' in s3_key:
                    asset_type = AssetType.IMAGE
                else:
                    asset_type = AssetType.IMAGE  # default
                
                # SHA256 계산
                sha256_hash = calculate_sha256_from_s3(s3_key)
                
                # 이미 존재하는 Asset 확인 (sha256 기준)
                existing_asset = db.query(Asset).filter(
                    Asset.sha256 == sha256_hash
                ).first()
                
                if existing_asset:
                    # 이미 존재하는 경우 기존 Asset 사용
                    asset = existing_asset
                    # storage_uri가 다를 수 있으므로 업데이트 (같은 파일이 다른 경로에 있을 수 있음)
                    if asset.storage_uri != s3_key:
                        asset.storage_uri = s3_key
                else:
                    # 새 Asset 생성
                    asset = Asset(
                        dataset_split_id=dataset_split.id,
                        type=asset_type,
                        storage_uri=s3_key,
                        sha256=sha256_hash,
                        bytes=item.file_size,
                        width=item.width,
                        height=item.height,
                        duration_ms=item.duration_ms if asset_type == AssetType.VIDEO else None,
                        fps=None,  # 비디오 메타데이터는 추후 확장
                        frame=None,
                        codec=None
                    )
                    db.add(asset)
                
                db.flush()  # ID를 즉시 생성하기 위해 flush
                
                successful.append({
                    "filename": item.original_filename,
                    "s3_key": s3_key,
                    "asset_id": asset.id,
                    "type": asset_type.value
                })
                
            except Exception as e:
                failed.append({
                    "filename": getattr(item, "original_filename", "unknown"),
                    "error": str(e)
                })
                print(f"[ERROR] Failed to process upload item: {e}")
                import traceback
                traceback.print_exc()
        
        # 변경사항 커밋
        db.commit()
        
        print(f"[UPLOAD] Successfully processed {len(successful)} assets for dataset {dataset_id}")
        
        return {
            "success": True,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful": successful,
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Batch upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process batch upload: {str(e)}"
        )


@router.get("/{dataset_id}/assets")
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
    # 데이터셋 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Asset 타입 변환
    asset_type = None
    if kind == "image":
        asset_type = AssetType.IMAGE
    elif kind == "video":
        asset_type = AssetType.VIDEO
    
    # Asset 조회
    assets, total_count = get_assets_from_dataset(
        db=db,
        dataset_id=dataset_id,
        asset_type=asset_type,
        page=page,
        limit=limit
    )
    
    # DatasetAssetResponse 형식으로 변환 (API 호환성)
    result = []
    for asset in assets:
        filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
        result.append({
            "id": asset.id,
            "dataset_id": dataset_id,
            "kind": asset.type.value,
            "original_filename": filename,
            "s3_key": asset.storage_uri,
            "file_size": asset.bytes,
            "width": asset.width,
            "height": asset.height,
            "duration_ms": asset.duration_ms,
            "created_at": asset.created_at.isoformat() if asset.created_at else None,
            "created_by": current_user["uid"]
        })
    
    return result


@router.get("/assets/{asset_id}/presigned-download")
async def get_asset_download_url(
    asset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """
    Asset 단건 다운로드용 Presigned URL 발급
    
    - 이미지/동영상 모두 지원
    - URL 만료 시간: 1시간
    """
    # Asset 조회
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # 데이터셋 권한 확인 (DatasetSplit -> DatasetVersion -> Dataset)
    dataset_split = asset.dataset_split
    if not dataset_split:
        raise HTTPException(status_code=404, detail="Dataset split not found")
    
    dataset_version = dataset_split.dataset_version
    if not dataset_version:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_version.dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # 파일명 추출
        filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
        
        # Presigned GET URL 생성
        url_data = presigned_url_service.generate_download_url(
            s3_key=asset.storage_uri,
            expiration=3600,
            filename=filename
        )
        
        return {
            "success": True,
            "asset_id": asset.id,
            "original_filename": filename,
            "kind": asset.type.value,
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
    Asset 다운로드용 Presigned URL 발급 (단일/배치 모두 처리)
    
    - 단일 파일: asset_ids에 1개만 전달
    - 여러 파일: asset_ids에 여러 개 전달
    - 클라이언트가 각 URL로 직접 S3에서 다운로드
    - URL 만료 시간: 1시간
    """
    # 데이터셋 확인
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Asset 조회
    assets = db.query(Asset).join(DatasetSplit).join(DatasetVersion).filter(
        Asset.id.in_(request.asset_ids),
        DatasetVersion.dataset_id == dataset_id
        ).all()
    
    if not assets:
        raise HTTPException(status_code=404, detail="No assets found")
    
    # Asset ID로 매핑
    asset_map = {asset.id: asset for asset in assets}
    
    # Presigned URL 생성
    urls = []
    failed = []
    
    for asset_id in request.asset_ids:
        asset = asset_map.get(asset_id)
        
        if not asset:
            failed.append({
                "asset_id": asset_id,
                "error": "Asset not found"
            })
            continue
        
        try:
            filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
            url_data = presigned_url_service.generate_download_url(
                s3_key=asset.storage_uri,
                filename=filename,
                expiration=3600  # 1시간
            )
            
            urls.append({
                "asset_id": asset.id,
                "type": asset.type.value,
                "download_url": url_data["download_url"],
                "filename": filename,
                "s3_key": asset.storage_uri,
                "expires_in": url_data["expires_in"],
                "generation_time": url_data["generation_time"]
            })
        except Exception as e:
            failed.append({
                "asset_id": asset_id,
                "error": str(e)
            })
    
    return {
        "success": True,
        "urls": urls,
        "total_count": len(urls),
        "failed_count": len(failed),
        "failed": failed
    }