from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class DatasetAsset(Base):
    """Store metadata for dataset files (images and videos)"""
    __tablename__ = "dataset_assets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True)
    
    kind = Column(String, index=True)  # "image" | "video"
    original_filename = Column(String)
    s3_key = Column(String, unique=True, index=True)
    
    file_size = Column(Integer)  # bytes
    
    # Image/video dimensions
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # for videos
    
    # Additional metadata (e.g., codec, fps, format)
    extra_meta = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, index=True)
    
    # Relationships
    dataset = relationship("Dataset", backref="assets")

