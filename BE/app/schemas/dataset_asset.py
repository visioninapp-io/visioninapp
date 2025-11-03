from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class DatasetAssetBase(BaseModel):
    original_filename: str
    s3_key: str
    kind: str  # "image" | "video"
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None


class DatasetAssetCreate(DatasetAssetBase):
    dataset_id: int
    created_by: str


class DatasetAssetResponse(DatasetAssetBase):
    id: int
    dataset_id: int
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True


class UploadCompleteItem(BaseModel):
    """Single file upload completion data"""
    original_filename: str
    s3_key: str
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None


class UploadCompleteBatchRequest(BaseModel):
    """Batch upload completion request"""
    items: List[UploadCompleteItem]

