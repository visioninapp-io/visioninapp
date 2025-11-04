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


# Dataset Version Schemas
class DatasetVersionCreate(BaseModel):
    """Create a new dataset version"""
    dataset_id: int
    name: str
    description: Optional[str] = None

    # Data split
    train_split: float = Field(0.7, ge=0, le=1)
    valid_split: float = Field(0.2, ge=0, le=1)
    test_split: float = Field(0.1, ge=0, le=1)

    # Preprocessing and augmentation
    preprocessing_config: Optional[PreprocessingConfig] = None
    augmentation_config: Optional[AugmentationConfig] = None


class DatasetVersionUpdate(BaseModel):
    """Update dataset version"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class DatasetVersionResponse(BaseModel):
    """Dataset version response"""
    id: int
    dataset_id: int
    version_number: int
    name: str
    description: Optional[str]

    # Split info
    train_split: float
    valid_split: float
    test_split: float

    # Configs
    preprocessing_config: Dict[str, Any]
    augmentation_config: Dict[str, Any]

    # Stats
    total_images: int
    train_images: int
    valid_images: int
    test_images: int

    # Status
    status: str
    generation_progress: int

    created_at: datetime
    completed_at: Optional[datetime]
    created_by: str

    class Config:
        from_attributes = True


# Export Job Schemas
class ExportJobCreate(BaseModel):
    """Create export job"""
    dataset_id: Optional[int] = None
    version_id: Optional[int] = None
    export_format: str  # yolov8, yolov5, coco, pascal_voc, tfrecord, csv
    include_images: bool = True


class ExportJobResponse(BaseModel):
    """Export job response"""
    id: int
    dataset_id: Optional[int]
    version_id: Optional[int]
    export_format: str
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

