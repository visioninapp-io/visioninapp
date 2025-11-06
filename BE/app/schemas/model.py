from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.model import ModelFramework, ModelStatus


class ModelBase(BaseModel):
    """Model 기본 스키마"""
    name: str
    task: Optional[str] = None
    description: Optional[str] = None


class ModelCreate(ModelBase):
    """Model 생성 스키마"""
    pass


class ModelUpdate(BaseModel):
    """Model 업데이트 스키마"""
    name: Optional[str] = None
    task: Optional[str] = None
    description: Optional[str] = None


class ModelResponse(ModelBase):
    """Model 응답 스키마"""
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """Model 목록 응답 스키마"""
    id: int
    name: str
    task: Optional[str]
    description: Optional[str]
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True
