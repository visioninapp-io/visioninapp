from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.model import ModelFramework, ModelStatus


class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    architecture: str


class ModelCreate(ModelBase):
    framework: ModelFramework = ModelFramework.PYTORCH
    version: str = "1.0"
    training_config: Optional[Dict[str, Any]] = None


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ModelStatus] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None


class ModelResponse(ModelBase):
    id: int
    framework: ModelFramework
    status: ModelStatus
    version: str
    file_path: Optional[str]
    file_size: Optional[int]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    map_50: Optional[float]
    map_50_95: Optional[float]
    training_config: Optional[Dict[str, Any]]
    hyperparameters: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    created_by: str

    class Config:
        from_attributes = True


class ModelConversionRequest(BaseModel):
    source_model_id: int
    target_framework: ModelFramework
    optimization_level: str = Field("balanced", description="speed, balanced, or size")
    precision: str = Field("FP16", description="FP32, FP16, or INT8")

    class Config:
        protected_namespaces = ()


class ModelConversionResponse(BaseModel):
    id: int
    source_model_id: int
    target_framework: ModelFramework
    status: str
    optimization_level: str
    precision: str
    output_file_path: Optional[str]
    output_file_size: Optional[int]
    conversion_log: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelListResponse(BaseModel):
    id: int
    name: str
    framework: ModelFramework
    status: ModelStatus
    version: str
    accuracy: Optional[float]
    file_size: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True
