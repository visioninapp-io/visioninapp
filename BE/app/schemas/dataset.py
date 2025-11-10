from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.dataset import GeometryType


class AnnotationCreate(BaseModel):
    """Annotation 생성 스키마"""
    asset_id: int = Field(..., description="대상 Asset ID")
    label_class_id: int = Field(..., description="라벨 클래스 ID")
    model_version_id: Optional[int] = Field(None, description="자동 생성 모델 버전 ID (선택)")
    geometry_type: GeometryType = Field(GeometryType.BBOX, description="형태 (bbox, polygon, keypoint)")
    geometry: Dict[str, Any] = Field(..., description="좌표/포인트 데이터")
    is_normalized: bool = Field(True, description="정규화 여부")
    source: Optional[str] = Field("human", description="생성 주석 (human, model)")
    confidence: float = Field(1.0, ge=0, le=1, description="모델 신뢰도")
    annotator_name: Optional[str] = Field("system", description="라벨러")

    class Config:
        protected_namespaces = ()


class LabelClassInfo(BaseModel):
    """Label class 정보 (nested)"""
    id: int
    display_name: str
    color: str
    shape_type: Optional[str]

    class Config:
        from_attributes = True


class AnnotationResponse(BaseModel):
    """Annotation 응답 스키마"""
    id: int
    asset_id: int
    label_class_id: int
    model_version_id: Optional[int]
    geometry_type: GeometryType
    geometry: Dict[str, Any]
    is_normalized: bool
    source: Optional[str]
    confidence: float
    annotator_name: str
    created_at: datetime

    # Nested label_class information
    label_class: Optional[LabelClassInfo] = None

    class Config:
        from_attributes = True
        protected_namespaces = ()


class DatasetBase(BaseModel):
    """Dataset 기본 스키마 (ERD 기준)"""
    name: str
    description: Optional[str] = None

    @field_validator("description", mode="before")
    def ensure_str(cls, v):
        if v is None:
            return None
        return str(v)


class DatasetCreate(DatasetBase):
    """Dataset 생성 스키마"""
    pass


class DatasetUpdate(BaseModel):
    """Dataset 수정 스키마"""
    name: Optional[str] = None
    description: Optional[str] = None

    @field_validator("description", mode="before")
    def ensure_str(cls, v):
        if v is None:
            return None
        return str(v)


class DatasetResponse(DatasetBase):
    """Dataset 응답 스키마 (ERD 기준 + 계산 필드)"""
    id: int
    project_id: int
    created_at: datetime
    
    # 계산된 필드들 (동적으로 계산)
    total_assets: Optional[int] = None  # Asset 개수
    total_images: Optional[int] = None  # 이미지 개수
    annotated_images: Optional[int] = None  # 어노테이션된 이미지 개수
    total_classes: Optional[int] = None  # 클래스 개수
    total_annotations: Optional[int] = None  # Annotation 개수
    version_count: Optional[int] = None  # 버전 개수

    class Config:
        from_attributes = True


class DatasetStats(BaseModel):
    """Dataset 통계"""
    total_assets: int  # 전체 Asset 개수 (이미지 + 비디오)
    total_images: int  # 이미지 개수
    total_datasets: int  # 데이터셋 개수
    total_annotations: int  # 총 어노테이션 개수
    total_classes: int  # 전체 클래스 개수
    auto_annotation_rate: int  # 어노테이션 완료율 (퍼센트)


class AutoAnnotationRequest(BaseModel):
    """자동 어노테이션 요청"""
    dataset_id: int
    model_id: Optional[int] = None  # Optional: use default model if not specified
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    overwrite_existing: bool = False

    class Config:
        protected_namespaces = ()
