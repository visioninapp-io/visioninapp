from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, BigInteger, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from app.utils.timezone import get_kst_now_naive


class ModelArtifact(Base):
    """Store metadata for model files (pt, onnx, trt, etc.)"""
    __tablename__ = "model_artifact"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=False, index=True, comment="소속 모델 버전ID")
    
    format = Column(Text, nullable=False, comment="형식")  # 'onnx', 'pth', 'tf'
    device = Column(Text, nullable=False, comment="실행 환경")  # 'cpu', 'gpu'
    precision = Column(Float, nullable=False, comment="정밀도")
    storage_uri = Column(Text, nullable=False, comment="파일 경로")
    sha256 = Column(String(64), nullable=False, unique=True, comment="무결성")
    size_bytes = Column(BigInteger, nullable=False, comment="파일 크기")
    opset = Column(Integer, nullable=True, comment="ONNX용 오프셋")
    ir_version = Column(Integer, nullable=True, comment="IR 버전")
    compat = Column(JSON, nullable=True, comment="환경 호환성")
    created_at = Column(DateTime, nullable=False, default=get_kst_now_naive, comment="생성일")
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="artifacts", foreign_keys=[model_version_id])

