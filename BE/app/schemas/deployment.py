from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.models.deployment import DeploymentTarget, DeploymentStatus


class DeploymentBase(BaseModel):
    """Deployment 기본 스키마"""
    name: str
    target: DeploymentTarget

    class Config:
        protected_namespaces = ()


class DeploymentCreate(DeploymentBase):
    """Deployment 생성 스키마"""
    model_version_id: int = Field(..., description="모델 버전 ID")
    endpoint_url: str = Field(..., description="엔드포인트 URL")


class DeploymentUpdate(BaseModel):
    """Deployment 업데이트 스키마"""
    name: Optional[str] = None
    endpoint_url: Optional[str] = None


class DeploymentResponse(DeploymentBase):
    """Deployment 응답 스키마"""
    id: int
    model_version_id: int
    endpoint_url: str
    deployed_at: datetime

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


class Config:
        from_attributes = True


class DeploymentStats(BaseModel):
    active_deployments: int
    total_requests: int
    avg_response_time: float
    uptime_percentage: float
