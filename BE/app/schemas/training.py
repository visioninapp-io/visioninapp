from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.training import TrainingStatus

class TrainingJobCreate(BaseModel):
    name: str
    dataset_id: int = Field(..., description="Dataset ID to train on")
    architecture: str = Field(..., description="Model architecture (e.g., 'yolov8n')")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=lambda: {"epochs": 20, "batch": 8, "imgsz": 640}
    )

class TrainingJobUpdate(BaseModel):
    status: Optional[TrainingStatus] = None
    s3_log_uri: Optional[str] = None
    error_message: Optional[str] = None

class TrainingJobResponse(BaseModel):
    id: int
    name: str
    architecture: Optional[str] = None
    model_id: Optional[int] = None
    dataset_id: Optional[int] = None

    hyperparameters: Dict[str, Any]
    status: TrainingStatus
    total_epochs: Optional[int] = None
    current_epoch: Optional[int] = None

    s3_log_uri: Optional[str] = None
    external_job_id: Optional[str] = None

    created_at: datetime
    created_by: str
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
        protected_namespaces = ()