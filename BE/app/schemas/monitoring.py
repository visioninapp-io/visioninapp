from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.models.monitoring import AlertSeverity, AlertStatus


class MonitoringAlertResponse(BaseModel):
    id: int
    deployment_id: Optional[int]
    model_id: Optional[int]
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    alert_type: str
    metric_value: Optional[float]
    threshold_value: Optional[float]
    alert_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]

    class Config:
        from_attributes = True
        protected_namespaces = ()


class AlertAcknowledgeRequest(BaseModel):
    alert_id: int


class PerformanceMetricResponse(BaseModel):
    id: int
    deployment_id: int
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    avg_latency: float
    requests_count: int
    error_count: int
    error_rate: float
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    gpu_usage: Optional[float]
    window_start: datetime
    window_end: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackLoopBase(BaseModel):
    name: str
    deployment_id: int


class FeedbackLoopCreate(FeedbackLoopBase):
    accuracy_threshold: float = Field(0.93, ge=0, le=1)
    data_collection_enabled: bool = True
    auto_retrain_enabled: bool = False
    retrain_schedule: Optional[str] = None
    min_samples_for_retrain: int = Field(1000, ge=0)
    configuration: Optional[Dict[str, Any]] = None


class FeedbackLoopUpdate(BaseModel):
    is_enabled: Optional[bool] = None
    accuracy_threshold: Optional[float] = None
    data_collection_enabled: Optional[bool] = None
    auto_retrain_enabled: Optional[bool] = None
    retrain_schedule: Optional[str] = None
    min_samples_for_retrain: Optional[int] = None
    configuration: Optional[Dict[str, Any]] = None


class FeedbackLoopResponse(FeedbackLoopBase):
    id: int
    is_enabled: bool
    accuracy_threshold: float
    data_collection_enabled: bool
    auto_retrain_enabled: bool
    retrain_schedule: Optional[str]
    min_samples_for_retrain: int
    collected_samples_count: int
    last_retrain_triggered: Optional[datetime]
    configuration: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EdgeCaseResponse(BaseModel):
    id: int
    deployment_id: int
    feedback_loop_id: Optional[int]
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, Any]
    is_low_confidence: bool
    is_misclassified: bool
    is_reviewed: bool
    ground_truth: Optional[Dict[str, Any]]
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class EdgeCaseReviewRequest(BaseModel):
    edge_case_id: int
    ground_truth: Dict[str, Any]


class MonitoringDashboardResponse(BaseModel):
    live_accuracy: float
    avg_latency: float
    today_inferences: int
    active_alerts: int
    recent_metrics: List[PerformanceMetricResponse]
    accuracy_trend: List[Dict[str, Any]]
    alerts: List[MonitoringAlertResponse]
