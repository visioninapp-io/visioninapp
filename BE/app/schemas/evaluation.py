from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ClassMetric(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class EvaluationCreate(BaseModel):
    """Evaluation 생성 스키마"""
    model_version_id: int = Field(..., description="모델 버전 ID")
    f1_score: float = Field(..., ge=0, le=1, description="조화평균")
    precision: float = Field(..., ge=0, le=1, description="정밀도")
    recall: float = Field(..., ge=0, le=1, description="재현율")
    mAP_50: float = Field(..., ge=0, le=1, description="평균정확도평균50")
    mAP_50_95: float = Field(..., ge=0, le=1, description="평균정확도평균95")

    class Config:
        protected_namespaces = ()


class EvaluationResponse(BaseModel):
    """Evaluation 응답 스키마"""
    id: int
    model_version_id: int
    f1_score: float
    precision: float
    recall: float
    mAP_50: float
    mAP_50_95: float

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelComparisonResponse(BaseModel):
    models: List[Dict[str, Any]]
    metrics: List[str]
    comparison_chart_data: Dict[str, Any]
