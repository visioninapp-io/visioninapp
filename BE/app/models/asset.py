from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Text, BigInteger, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class AssetType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"


class Asset(Base):
    """데이터셋 스플릿 내의 개별 데이터 자산(예: 이미지 파일, 비디오 파일) 정보"""
    __tablename__ = "asset"

    id = Column(Integer, primary_key=True, index=True)
    dataset_split_id = Column(Integer, ForeignKey("dataset_split.id"), nullable=False, index=True, comment="스플릿ID")
    name = Column(String(255), nullable=False, comment="에셋명")
    type = Column(Enum(AssetType), nullable=False, comment="타입")
    storage_uri = Column(Text, nullable=False, comment="파일 경로")
    sha256 = Column(String(64), nullable=False, comment="무결성")
    bytes = Column(BigInteger, nullable=True, comment="파일 크기")
    
    # 이미지 관련 필드
    width = Column(Integer, nullable=True, comment="이미지 폭")
    height = Column(Integer, nullable=True, comment="이미지 높이")
    
    # 비디오 관련 필드
    duration_ms = Column(Integer, nullable=True, comment="비디오 길이")
    fps = Column(Float, nullable=True, comment="비디오 FPS")
    frame = Column(Integer, nullable=True, comment="비디오 프레임")
    codec = Column(Text, nullable=True, comment="비디오 코덱")
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성일")

    # Relationships
    dataset_split = relationship("DatasetSplit", back_populates="assets")
    annotations = relationship("Annotation", back_populates="asset", cascade="all, delete-orphan")

