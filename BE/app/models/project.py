from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class Project(Base):
    """프로젝트 정보 - 데이터셋, 모델, 레이블 온톨로지 등 관련 리소스를 그룹화"""
    __tablename__ = "project"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, comment="프로젝트명")
    description = Column(Text, nullable=True, comment="설명")
    created_at = Column(DateTime, nullable=True, default=get_kst_now_naive, comment="생성일")

    # Relationships
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="project", cascade="all, delete-orphan")