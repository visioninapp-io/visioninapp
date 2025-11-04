from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class DatasetStatus(str, enum.Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    ANNOTATED = "annotated"
    READY = "ready"
    ERROR = "error"


class DatasetType(str, enum.Enum):
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    dataset_type = Column(Enum(DatasetType), default=DatasetType.OBJECT_DETECTION)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.CREATED)

    total_images = Column(Integer, default=0)
    annotated_images = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)
    class_names = Column(JSON, default=lambda: [])
    class_colors = Column(JSON, default=lambda: {})  # {class_name: color_hex}

    # Auto-annotation settings
    auto_annotation_enabled = Column(Integer, default=0)  # boolean as int for SQLite
    auto_annotation_model_id = Column(Integer, ForeignKey("models.id"), nullable=True)

    # Project settings
    is_public = Column(Integer, default=0)  # Public datasets can be shared

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    images = relationship("Image", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    batches = relationship("UploadBatch", back_populates="dataset", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)

    is_annotated = Column(Integer, default=0)  # boolean as int
    annotation_data = Column(JSON, nullable=True)  # YOLO/COCO format annotations

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="images")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))

    class_id = Column(Integer)
    class_name = Column(String)

    # Bounding box coordinates (normalized 0-1)
    x_center = Column(Float)
    y_center = Column(Float)
    width = Column(Float)
    height = Column(Float)

    # Polygon points for segmentation (JSON array of [x, y] normalized coordinates)
    polygon_points = Column(JSON, nullable=True)

    confidence = Column(Float, default=1.0)
    is_auto_generated = Column(Integer, default=0)  # boolean as int
    is_verified = Column(Integer, default=0)  # boolean as int

    created_at = Column(DateTime, default=datetime.utcnow)


class DatasetVersion(Base):
    """Dataset version with preprocessing and augmentation settings"""
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    version_number = Column(Integer)  # 1, 2, 3, etc.
    name = Column(String)  # e.g., "v1-augmented", "v2-grayscale"
    description = Column(Text, nullable=True)

    # Data split ratios
    train_split = Column(Float, default=0.7)
    valid_split = Column(Float, default=0.2)
    test_split = Column(Float, default=0.1)

    # Preprocessing settings (JSON)
    preprocessing_config = Column(JSON, default=dict)

    # Augmentation settings (JSON)
    augmentation_config = Column(JSON, default=dict)

    # Stats after generation
    total_images = Column(Integer, default=0)
    train_images = Column(Integer, default=0)
    valid_images = Column(Integer, default=0)
    test_images = Column(Integer, default=0)

    # Generation status
    status = Column(String, default="pending")  # pending, generating, completed, failed
    generation_progress = Column(Integer, default=0)  # 0-100

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String)

    # Relationships
    dataset = relationship("Dataset", back_populates="versions")


class UploadBatch(Base):
    """Track batch uploads and their status"""
    __tablename__ = "upload_batches"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    batch_name = Column(String)
    total_files = Column(Integer, default=0)
    successful_uploads = Column(Integer, default=0)
    failed_uploads = Column(Integer, default=0)

    status = Column(String, default="uploading")  # uploading, completed, failed
    error_messages = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String)

    # Relationships
    dataset = relationship("Dataset", back_populates="batches")


class ExportJob(Base):
    """Track dataset export jobs"""
    __tablename__ = "export_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)

    export_format = Column(String)  # yolov8, yolov5, coco, pascal_voc, etc.
    include_images = Column(Integer, default=1)  # boolean as int

    # File info
    file_path = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    download_url = Column(String, nullable=True)

    status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String)
