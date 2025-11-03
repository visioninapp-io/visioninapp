from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class ModelArtifact(Base):
    """Store metadata for model files (pt, onnx, trt, etc.)"""
    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), index=True)
    
    kind = Column(String, index=True)  # "pt" | "onnx" | "trt" | "tflite"
    version = Column(String, default="1.0")
    
    s3_key = Column(String, unique=True, index=True)
    file_size = Column(Integer)  # bytes
    
    checksum = Column(String, nullable=True)  # sha256 hash
    is_primary = Column(Boolean, default=True)  # main artifact for this model
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, index=True)
    
    # Relationships
    model = relationship("Model", backref="artifacts")

