from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class LabelOntologyVersion(Base):
    """레이블 온톨로지의 버전을 관리"""
    __tablename__ = "label_ontology_version"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False, index=True, comment="프로젝트ID")
    version_tag = Column(String(50), nullable=False, comment="버전명")
    is_frozen = Column(Boolean, nullable=False, default=False, comment="수정 불가 여부")
    created_at = Column(DateTime, nullable=False, default=get_kst_now_naive, comment="생성일")

    # Relationships
    project = relationship("Project", back_populates="label_ontology_versions")
    label_classes = relationship("LabelClass", back_populates="ontology_version", cascade="all, delete-orphan")
    dataset_versions = relationship("DatasetVersion", back_populates="ontology_version")

