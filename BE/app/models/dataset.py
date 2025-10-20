from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
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


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.CREATED)

    total_images = Column(Integer, default=0)
    annotated_images = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)
    class_names = Column(JSON, default=list)

    auto_annotation_enabled = Column(Integer, default=0)  # boolean as int for SQLite
    auto_annotation_model_id = Column(Integer, ForeignKey("models.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    images = relationship("Image", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset")


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

    # Bounding box coordinates (normalized)
    x_center = Column(Float)
    y_center = Column(Float)
    width = Column(Float)
    height = Column(Float)

    confidence = Column(Float, default=1.0)
    is_auto_generated = Column(Integer, default=0)  # boolean as int
    is_verified = Column(Integer, default=0)  # boolean as int

    created_at = Column(DateTime, default=datetime.utcnow)
