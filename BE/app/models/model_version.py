from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class ModelVersion(Base):
    """모델의 특정 버전에 대한 정보를 관리"""
    __tablename__ = "model_version"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model.id"), nullable=False, index=True, comment="상위 모델ID")
    dataset_version_id = Column(Integer, ForeignKey("dataset_version.id"), nullable=False, index=True, comment="데이터셋 버전ID")
    parent_model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=True, index=True, comment="증분학습 부모")
    init_from_artifact_id = Column(Integer, ForeignKey("model_artifact.id"), nullable=True, index=True, comment="초기 가중치")
    
    framework = Column(String(50), nullable=False, comment="라이브러리")  # 'pytorch', 'tensorflow'
    framework_version = Column(String(50), nullable=False, comment="라이브러리 버전")
    training_config = Column(JSON, nullable=False, comment="하이퍼파라미터")
    incremental_info = Column(JSON, nullable=False, comment="증분학습 전략")
    is_frozen = Column(Boolean, nullable=False, default=False, comment="수정 불가 여부")
    created_at = Column(DateTime, nullable=False, default=get_kst_now_naive, comment="생성일")

    # Relationships
    model = relationship("Model", back_populates="versions", foreign_keys=[model_id])
    dataset_version = relationship("DatasetVersion", back_populates="model_versions")
    parent_model_version = relationship("ModelVersion", remote_side=[id], backref="child_versions")
    init_from_artifact = relationship("ModelArtifact", foreign_keys=[init_from_artifact_id])
    
    artifacts = relationship(
        "ModelArtifact", 
        back_populates="model_version", 
        cascade="all, delete-orphan",
        foreign_keys="[ModelArtifact.model_version_id]"
    )
    evaluations = relationship("Evaluation", back_populates="model_version", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="model_version", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="model_version")

