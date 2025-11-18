from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    __tablename__ = "training_job"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True)

    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    model_id = Column(Integer, ForeignKey("model.id", ondelete="CASCADE"), nullable=True)

    status = Column(String(50), default=TrainingStatus.PENDING.value)

    # Training configuration
    architecture = Column(String(100))  # YOLOv8, Faster R-CNN, etc.
    hyperparameters = Column(JSON)  # epochs, batch_size, learning_rate, etc.

    # Progress tracking
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    progress_percentage = Column(Float, default=0.0)

    # Real-time metrics (latest values)
    current_loss = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    current_learning_rate = Column(Float, nullable=True)

    # Training history (time series data)
    metrics_history = Column(JSON, default=dict)  # {epoch: {loss, accuracy, val_loss, val_accuracy}}

    # Time estimates
    started_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Logs and errors
    training_log = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=get_kst_now_naive)
    created_by = Column(String(100), index=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job")
