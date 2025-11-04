from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class LabelClass(Base):
    """특정 온톨로지 버전 내의 개별 레이블 클래스를 정의"""
    __tablename__ = "label_class"

    id = Column(Integer, primary_key=True, index=True)
    ontology_version_id = Column(Integer, ForeignKey("label_ontology_version.id"), nullable=False, index=True, comment="온톨로지 버전 ID")
    display_name = Column(String(100), nullable=False, comment="라벨이름")
    shape_type = Column(Text, nullable=True, comment="형태")  # 'bbox', 'polygon', 'keypoint' 등
    color = Column(String(20), nullable=False, comment="표시색상")
    keypoint_spec = Column(JSON, nullable=True, comment="포즈스켈레톤정의")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성일")

    # Relationships
    ontology_version = relationship("LabelOntologyVersion", back_populates="label_classes")
    annotations = relationship("Annotation", back_populates="label_class")

