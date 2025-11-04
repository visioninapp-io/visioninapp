from sqlalchemy import Column, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.core.database import Base


class Evaluation(Base):
    __tablename__ = "evaluation"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=False, index=True, comment="모델 버전ID")
    
    f1_score = Column(Float, nullable=False, comment="조화평균")
    precision = Column(Float, nullable=False, comment="정밀도")
    recall = Column(Float, nullable=False, comment="재현율")
    mAP_50 = Column(Float, nullable=False, comment="평균정확도평균50")
    mAP_50_95 = Column(Float, nullable=False, comment="평균정확도평균95")

    # Relationships
    model_version = relationship("ModelVersion", back_populates="evaluations")
