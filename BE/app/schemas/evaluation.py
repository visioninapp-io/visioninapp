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
    model_id: int
    name: str
    description: Optional[str] = None
    test_dataset_name: Optional[str] = None
    test_dataset_size: int

    class Config:
        protected_namespaces = ()


class EvaluationResponse(BaseModel):
    id: int
    model_id: int
    name: str
    description: Optional[str]
    precision: float
    recall: float
    f1_score: float
    map_50: float
    map_50_95: Optional[float]
    class_metrics: List[Dict[str, Any]]
    confusion_matrix: Optional[Dict[str, Any]]
    test_dataset_size: int
    test_dataset_name: Optional[str]
    comparison_data: Optional[Dict[str, Any]]
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelComparisonResponse(BaseModel):
    models: List[Dict[str, Any]]
    metrics: List[str]
    comparison_chart_data: Dict[str, Any]
