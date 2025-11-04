from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ModelArtifactBase(BaseModel):
    kind: str  # "pt" | "onnx" | "trt" | "tflite"
    version: str = "1.0"
    s3_key: str
    file_size: int
    checksum: Optional[str] = None
    is_primary: bool = True


class ModelArtifactCreate(ModelArtifactBase):
    model_id: int
    created_by: str

    class Config:
        protected_namespaces = ()


class ModelArtifactResponse(ModelArtifactBase):
    id: int
    model_id: int
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelUploadRequest(BaseModel):
    """Request for model artifact presigned upload URL"""
    filename: str
    content_type: Optional[str] = "application/octet-stream"


class ModelUploadCompleteRequest(BaseModel):
    """Model upload completion data"""
    s3_key: str
    file_size: int
    checksum: Optional[str] = None
    kind: str = "pt"  # file type
    version: str = "1.0"
    is_primary: bool = True

