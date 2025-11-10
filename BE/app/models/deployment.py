from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


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
    __tablename__ = "deployment"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=False, index=True, comment="모델 버전ID")
    
    name = Column(String(50), nullable=False, comment="이름")
    target = Column(Enum(DeploymentTarget), nullable=False, comment="배포위치")  # 'edge', 'cloud'
    endpoint_url = Column(String(100), nullable=False, comment="엔드포인트")
    deployed_at = Column(DateTime, nullable=False, default=get_kst_now_naive, comment="배포일")

    # Relationships
    model_version = relationship("ModelVersion", back_populates="deployments")
