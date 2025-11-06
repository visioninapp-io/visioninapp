from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.training import TrainingStatus

class TrainingJobCreate(BaseModel):
    name: str
    # 실제 학습 모델은 hyperparameters["model"]이 우선, architecture는 조회용/백업용
    architecture: Optional[str] = Field(None, description="Optional; hyperparams.model is the source of truth")
    dataset_name: str = Field(..., description="e.g., 'myset'")
    dataset_s3_prefix: str = Field(..., description="e.g., 'datasets/myset/' (trailing slash recommended)")
    hyperparameters: Dict[str, Any] = Field(
        default={"model": "yolov8n", "epochs": 20, "batch": 8, "imgsz": 640}
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

    dataset_name: str
    dataset_s3_prefix: str
    hyperparameters: Dict[str, Any]
    status: TrainingStatus

    s3_log_uri: Optional[str] = None
    external_job_id: Optional[str] = None

    created_at: datetime
    created_by: str
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
        protected_namespaces = ()