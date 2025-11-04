from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class ModelFramework(str, enum.Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CONVERTING = "converting"
    READY = "ready"


class Model(Base):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False, index=True, comment="프로젝트ID")
    name = Column(String(255), nullable=False, comment="모델명")
    task = Column(Text, nullable=True, comment="작업")  # 'object_detection', 'segmentation'
    description = Column(Text, nullable=True, comment="설명")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성일")

    # Relationships
    project = relationship("Project", back_populates="models")
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan", foreign_keys="ModelVersion.model_id")
    training_job = relationship("TrainingJob", back_populates="model", uselist=False)
