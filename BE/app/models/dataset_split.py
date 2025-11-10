from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class DatasetSplitType(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNASSIGNED = "unassigned"


class DatasetSplit(Base):
    """데이터셋 버전을 훈련(train), 검증(val), 테스트(test) 등으로 분할한 정보"""
    __tablename__ = "dataset_split"

    id = Column(Integer, primary_key=True, index=True)
    dataset_version_id = Column(Integer, ForeignKey("dataset_version.id"), nullable=False, index=True, comment="데이터셋 버전ID")
    split = Column(Enum(DatasetSplitType), nullable=False, comment="데이터 구분")
    ratio = Column(Float, nullable=False, comment="비율")
    created_at = Column(DateTime, nullable=False, default=get_kst_now_naive, comment="생성일")

    # Relationships
    dataset_version = relationship("DatasetVersion", back_populates="splits")
    assets = relationship("Asset", back_populates="dataset_split", cascade="all, delete-orphan")

