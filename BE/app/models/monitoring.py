from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)

    severity = Column(Enum(AlertSeverity))
    status = Column(Enum(AlertStatus), default=AlertStatus.ACTIVE)

    title = Column(String)
    message = Column(Text)

    alert_type = Column(String)  # accuracy_drop, latency_spike, error_rate, etc.
    metric_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)

    alert_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    acknowledged_by = Column(String, nullable=True)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))

    # Accuracy metrics (from live inference)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)

    # Performance metrics
    avg_latency = Column(Float)  # milliseconds
    requests_count = Column(Integer)
    error_count = Column(Integer, default=0)
    error_rate = Column(Float, default=0.0)

    # Resource utilization
    cpu_usage = Column(Float, nullable=True)
    memory_usage = Column(Float, nullable=True)
    gpu_usage = Column(Float, nullable=True)

    # Time window
    window_start = Column(DateTime)
    window_end = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class FeedbackLoop(Base):
    __tablename__ = "feedback_loops"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))

    name = Column(String)
    is_enabled = Column(Integer, default=1)  # boolean as int

    # Triggers
    accuracy_threshold = Column(Float, default=0.93)
    data_collection_enabled = Column(Integer, default=1)
    auto_retrain_enabled = Column(Integer, default=0)

    # Scheduling
    retrain_schedule = Column(String, nullable=True)  # cron expression
    min_samples_for_retrain = Column(Integer, default=1000)

    # Collected data
    collected_samples_count = Column(Integer, default=0)
    last_retrain_triggered = Column(DateTime, nullable=True)

    # Configuration
    configuration = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EdgeCase(Base):
    __tablename__ = "edge_cases"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))
    feedback_loop_id = Column(Integer, ForeignKey("feedback_loops.id"), nullable=True)

    image_data = Column(Text)  # Base64 or file path
    predictions = Column(JSON)
    confidence_scores = Column(JSON)

    is_low_confidence = Column(Integer, default=0)
    is_misclassified = Column(Integer, default=0)
    is_reviewed = Column(Integer, default=0)

    ground_truth = Column(JSON, nullable=True)  # manual annotation
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
