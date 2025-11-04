from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.models.deployment import DeploymentTarget, DeploymentStatus


class DeploymentBase(BaseModel):
    name: str
    model_id: int
    target: DeploymentTarget
    device_type: str

    class Config:
        protected_namespaces = ()


class DeploymentCreate(DeploymentBase):
    configuration: Optional[Dict[str, Any]] = None


class DeploymentUpdate(BaseModel):
    status: Optional[DeploymentStatus] = None
    endpoint_url: Optional[str] = None
    total_requests: Optional[int] = None
    requests_today: Optional[int] = None
    avg_response_time: Optional[float] = None
    uptime_percentage: Optional[float] = None
    health_status: Optional[str] = None


class DeploymentResponse(DeploymentBase):
    id: int
    status: DeploymentStatus
    endpoint_url: Optional[str]
    api_key: Optional[str]
    configuration: Optional[Dict[str, Any]]
    total_requests: int
    requests_today: int
    avg_response_time: Optional[float]
    uptime_percentage: Optional[float]
    last_health_check: Optional[datetime]
    health_status: str
    deployed_at: Optional[datetime]
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class InferenceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image or image URL")
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    return_visualization: bool = Field(False)


class PredictionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, float] = Field(..., description="Bounding box: {x, y, width, height}")


class InferenceResponse(BaseModel):
    request_id: str
    predictions: List[PredictionResult]
    inference_time: float
    preprocessing_time: Optional[float]
    postprocessing_time: Optional[float]
    visualization_url: Optional[str] = None


class InferenceLogResponse(BaseModel):
    id: int
    deployment_id: int
    request_id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: Optional[Dict[str, Any]]
    inference_time: float
    preprocessing_time: Optional[float]
    postprocessing_time: Optional[float]
    timestamp: datetime

    class Config:
        from_attributes = True


class DeploymentStats(BaseModel):
    active_deployments: int
    total_requests: int
    avg_response_time: float
    uptime_percentage: float
