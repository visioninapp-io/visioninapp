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
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)

    framework = Column(Enum(ModelFramework), default=ModelFramework.PYTORCH)
    status = Column(Enum(ModelStatus), default=ModelStatus.TRAINING)

    version = Column(String, default="1.0")
    architecture = Column(String)  # YOLOv8, Faster R-CNN, etc.

    file_path = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)  # in bytes

    # Training metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    map_50 = Column(Float, nullable=True)  # mAP@0.5
    map_50_95 = Column(Float, nullable=True)  # mAP@0.5:0.95

    # Training configuration
    training_config = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="model", uselist=False)
    conversions = relationship("ModelConversion", back_populates="source_model")
    deployments = relationship("Deployment", back_populates="model")
    evaluations = relationship("Evaluation", back_populates="model")


class ModelConversion(Base):
    __tablename__ = "model_conversions"

    id = Column(Integer, primary_key=True, index=True)
    source_model_id = Column(Integer, ForeignKey("models.id"))

    target_framework = Column(Enum(ModelFramework))
    status = Column(String, default="pending")  # pending, converting, completed, failed

    optimization_level = Column(String)  # speed, balanced, size
    precision = Column(String)  # FP32, FP16, INT8

    output_file_path = Column(String, nullable=True)
    output_file_size = Column(Integer, nullable=True)

    conversion_log = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    source_model = relationship("Model", back_populates="conversions")
