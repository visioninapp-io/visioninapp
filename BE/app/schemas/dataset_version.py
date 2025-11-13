from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Preprocessing Configuration Schema
class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for dataset version"""
    # Resize settings
    resize: Optional[Dict[str, Any]] = None  # {"width": 640, "height": 640, "mode": "fit"}  # fit, stretch, pad

    # Image transformations
    auto_orient: bool = True
    grayscale: bool = False

    # Normalization
    normalize_contrast: bool = False
    normalize_exposure: bool = False

    # Filtering
    filter_null_images: bool = True  # Remove images with no annotations
    min_box_size: Optional[int] = None  # Minimum bounding box size in pixels

    # Cropping
    static_crop: Optional[Dict[str, int]] = None  # {"top": 0, "bottom": 0, "left": 0, "right": 0}
    tile_images: Optional[Dict[str, Any]] = None  # {"rows": 2, "cols": 2, "overlap": 0}

    # Class/Tag filtering
    include_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None


# Augmentation Configuration Schema
class AugmentationConfig(BaseModel):
    """Augmentation configuration for dataset version"""
    # Number of augmented outputs per image
    output_count: int = Field(1, ge=1, le=10)

    # Geometric transformations
    flip_horizontal: Optional[float] = None  # Probability 0-1
    flip_vertical: Optional[float] = None
    rotate: Optional[Dict[str, Any]] = None  # {"min": -15, "max": 15}
    crop: Optional[Dict[str, Any]] = None  # {"min": 0, "max": 20}  # percentage
    shear: Optional[Dict[str, Any]] = None  # {"horizontal": 15, "vertical": 15}  # degrees

    # Color adjustments
    hue: Optional[Dict[str, int]] = None  # {"min": -25, "max": 25}
    saturation: Optional[Dict[str, int]] = None  # {"min": -25, "max": 25}
    brightness: Optional[Dict[str, int]] = None  # {"min": -25, "max": 25}
    exposure: Optional[Dict[str, int]] = None  # {"min": -25, "max": 25}

    # Noise and effects
    blur: Optional[Dict[str, float]] = None  # {"max": 2.5}  # pixels
    noise: Optional[Dict[str, int]] = None  # {"max": 5}  # percentage

    # Advanced augmentations
    cutout: Optional[Dict[str, Any]] = None  # {"count": 3, "size": 20}  # percentage
    mosaic: Optional[bool] = None  # Combine 4 images
    mixup: Optional[float] = None  # Alpha value for mixup


# Dataset Version Schemas (ERD 구조 기준)
class DatasetVersionCreate(BaseModel):
    """Create a new dataset version"""
    version_tag: Optional[str] = None  # None이면 자동 생성 (v1.0, v2.0, ...)
    is_frozen: bool = False


class DatasetVersionUpdate(BaseModel):
    """Update dataset version"""
    is_frozen: Optional[bool] = None


class DatasetSplitInfo(BaseModel):
    """Split information"""
    split: str  # train, val, test, unassigned
    ratio: float
    asset_count: int


class DatasetVersionResponse(BaseModel):
    """Dataset version response (ERD 구조)"""
    id: int
    dataset_id: int
    ontology_version_id: int
    version_tag: str
    is_frozen: bool
    created_at: datetime
    
    # Computed fields (조회 시 계산)
    splits: List[DatasetSplitInfo] = []
    total_assets: int = 0

    class Config:
        from_attributes = True


# Export Job Schemas
class ExportJobCreate(BaseModel):
    """Create export job"""
    dataset_id: Optional[int] = None
    version_id: Optional[int] = None
    include_images: bool = True


class ExportJobResponse(BaseModel):
    """Export job response"""
    id: int
    dataset_id: Optional[int]
    version_id: Optional[int]
    include_images: bool

    file_path: Optional[str]
    file_size: Optional[int]
    download_url: Optional[str]

    status: str
    error_message: Optional[str]

    created_at: datetime
    completed_at: Optional[datetime]
    created_by: str

    class Config:
        from_attributes = True

