from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.training import TrainingStatus


class TrainingJobBase(BaseModel):
    name: str
    dataset_id: int
    architecture: str = Field(..., description="Model architecture (e.g., YOLOv8, Faster R-CNN)")


class TrainingJobCreate(TrainingJobBase):
    hyperparameters: Dict[str, Any] = Field(
        default={
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "img_size": 640
        }
    )


class TrainingJobUpdate(BaseModel):
    status: Optional[TrainingStatus] = None
    current_epoch: Optional[int] = None
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    progress_percentage: Optional[float] = None


class TrainingJobResponse(TrainingJobBase):
    id: int
    model_id: Optional[int]
    status: TrainingStatus
    architecture: str
    hyperparameters: Dict[str, Any]
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    current_loss: Optional[float]
    current_accuracy: Optional[float]
    current_learning_rate: Optional[float]
    metrics_history: Dict[str, Any]
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    completed_at: Optional[datetime]
    training_log: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class TrainingControlRequest(BaseModel):
    action: str = Field(..., description="Action to perform: pause, resume, cancel")


class HyperparameterTuningRequest(BaseModel):
    dataset_id: int
    architecture: str
    search_space: Dict[str, Any] = Field(
        default={
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [8, 16, 32],
            "optimizer": ["Adam", "SGD"]
        }
    )
    n_trials: int = Field(10, ge=1, le=100)
