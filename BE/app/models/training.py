from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)

    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING)

    # Training configuration
    architecture = Column(String)  # YOLOv8, Faster R-CNN, etc.
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

    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))

    epoch = Column(Integer)
    step = Column(Integer, nullable=True)

    # Loss metrics
    train_loss = Column(Float)
    val_loss = Column(Float, nullable=True)

    # Performance metrics
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)

    # Learning rate
    learning_rate = Column(Float, nullable=True)

    # Additional metrics
    other_metrics = Column(JSON, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow)
