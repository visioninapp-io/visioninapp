from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class DeploymentTarget(str, enum.Enum):
    EDGE = "edge"
    CLOUD = "cloud"
    ON_PREMISE = "on_premise"
    MOBILE = "mobile"


class DeploymentStatus(str, enum.Enum):
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    model_id = Column(Integer, ForeignKey("models.id"))

    target = Column(Enum(DeploymentTarget))
    status = Column(Enum(DeploymentStatus), default=DeploymentStatus.DEPLOYING)

    # Deployment configuration
    device_type = Column(String)  # NVIDIA Jetson AGX, AWS Lambda, etc.
    endpoint_url = Column(String, nullable=True)
    api_key = Column(String, nullable=True)

    configuration = Column(JSON, nullable=True)  # deployment-specific settings

    # Performance metrics
    total_requests = Column(Integer, default=0)
    requests_today = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)  # milliseconds
    uptime_percentage = Column(Float, nullable=True)

    # Monitoring
    last_health_check = Column(DateTime, nullable=True)
    health_status = Column(String, default="unknown")

    deployed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    model = relationship("Model", back_populates="deployments")
    inference_logs = relationship("InferenceLog", back_populates="deployment")


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))

    # Request info
    request_id = Column(String, unique=True, index=True)
    image_data = Column(Text, nullable=True)  # Base64 or URL

    # Response info
    predictions = Column(JSON)  # [{class, confidence, bbox}, ...]
    confidence_scores = Column(JSON, nullable=True)

    # Performance
    inference_time = Column(Float)  # milliseconds
    preprocessing_time = Column(Float, nullable=True)
    postprocessing_time = Column(Float, nullable=True)

    # Metadata
    client_info = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    deployment = relationship("Deployment", back_populates="inference_logs")
