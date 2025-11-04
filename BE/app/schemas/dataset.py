from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.dataset import DatasetStatus, DatasetType


class ImageBase(BaseModel):
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None


class ImageCreate(ImageBase):
    dataset_id: int
    file_path: str
    file_size: int


class ImageResponse(ImageBase):
    id: int
    dataset_id: int
    file_path: str
    file_size: int
    is_annotated: bool
    annotation_data: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnnotationCreate(BaseModel):
    image_id: int
    class_id: int
    class_name: str
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)
    polygon_points: Optional[List[List[float]]] = None  # [[x1, y1], [x2, y2], ...]
    confidence: float = Field(1.0, ge=0, le=1)
    is_auto_generated: bool = False


class AnnotationResponse(BaseModel):
    id: int
    image_id: int
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    polygon_points: Optional[List[List[float]]] = None
    confidence: float
    is_auto_generated: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: Optional[DatasetType] = DatasetType.OBJECT_DETECTION


class DatasetCreate(DatasetBase):
    total_classes: Optional[int] = 0
    class_names: Optional[List[str]] = []
    class_colors: Optional[Dict[str, str]] = {}
    status: Optional[str] = "created"
    is_public: Optional[bool] = False


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None
    class_names: Optional[List[str]] = None
    auto_annotation_enabled: Optional[bool] = None
    auto_annotation_model_id: Optional[int] = None


class DatasetResponse(DatasetBase):
    id: int
    dataset_type: DatasetType
    status: DatasetStatus
    total_images: int
    annotated_images: int
    total_classes: int
    class_names: List[str]
    class_colors: Dict[str, str]
    auto_annotation_enabled: bool
    auto_annotation_model_id: Optional[int]
    is_public: bool
    created_at: datetime
    updated_at: datetime
    created_by: str

    class Config:
        from_attributes = True


class DatasetStats(BaseModel):
    total_images: int
    total_datasets: int
    total_classes: int
    auto_annotation_rate: float


class AutoAnnotationRequest(BaseModel):
    dataset_id: int
    model_id: Optional[int] = None  # Optional: use default model if not specified
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    overwrite_existing: bool = False

    class Config:
        protected_namespaces = ()
