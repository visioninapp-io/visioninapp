from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.config import settings
from app.models.model_artifact import ModelArtifact
from app.models.model_version import ModelVersion
from app.rabbitmq.producer import send_onnx_request, send_trt_request

router = APIRouter()

# Pydantic Models for Request Validation
class ModelInfo(BaseModel):
    s3_uri: str = Field(..., description="S3 URI of the model file (e.g., s3://bucket/path/model.pt)")

class OutputInfo(BaseModel):
    prefix: str = Field(..., description="S3 prefix for output file")
    model_name: Optional[str] = Field(None, description="Output model filename")

class OnnxOps(BaseModel):
    dynamic: bool = Field(True, description="Enable dynamic input shapes")
    simplify: bool = Field(True, description="Simplify ONNX model")
    opset: int = Field(13, description="ONNX opset version", ge=11, le=17)
    imgsz: int = Field(640, description="Input image size", gt=0)
    precision: Literal["fp32", "fp16", "int8"] = Field("fp32", description="Precision mode: fp32, fp16, or int8")

class TrtConfig(BaseModel):
    precision: Literal["fp32", "fp16", "int8"] = Field("fp16", description="Precision mode: fp32, fp16, or int8")
    imgsz: int = Field(640, description="Input image size", gt=0)
    dynamic: bool = Field(True, description="Enable dynamic input shapes")

class OnnxConversionRequest(BaseModel):
    job_id: Optional[str] = Field(None, description="Optional job ID")
    model: ModelInfo
    output: OutputInfo
    ops: Optional[OnnxOps] = Field(None, description="ONNX conversion options")

class TrtConversionRequest(BaseModel):
    job_id: Optional[str] = Field(None, description="Optional job ID")
    model: ModelInfo
    output: OutputInfo
    trt: Optional[TrtConfig] = Field(None, description="TensorRT conversion options")

def _job_id(s: Optional[str] = None) -> str:
    return s if s else str(uuid.uuid4()).replace("-", "")

def _key_from_s3_uri(s3_uri: str) -> str:
    # "s3://bucket/key/with/path.ext" -> "key/with/path.ext"
    if not s3_uri.startswith("s3://"):
        raise ValueError("invalid s3 uri")
    parts = s3_uri.split("/", 3)
    if len(parts) < 4:
        return ""
    return parts[3]

def _ensure_child_artifact(db: Session, *, src_key: str, out_prefix: str, out_name: str, out_format: str) -> ModelArtifact:
    """
    src_key 로 기존 아티팩트(PT)를 찾고, 그 아티팩트의 model_version_id(v0)에
    새로운 변환 아티팩트를 미리 INSERT(선점) 후 반환.
    """
    # 1) 원본 artifact 찾기
    src_art = db.query(ModelArtifact).filter(ModelArtifact.storage_uri == src_key).first()
    if not src_art:
        raise HTTPException(status_code=404, detail="source artifact not found")

    mv_id = src_art.model_version_id

    # 2) 새 artifact의 storage_uri 계산
    out_key = f"{out_prefix.rstrip('/')}/{out_name}"

    # 3) 이미 있으면 재사용, 없으면 선점 INSERT
    dst = db.query(ModelArtifact).filter(
        ModelArtifact.model_version_id == mv_id,
        ModelArtifact.storage_uri == out_key
    ).first()
    if dst:
        return dst

    dst = ModelArtifact(
        model_version_id=mv_id,
        storage_uri=out_key,
        format=out_format  # 참고용, 나머지 메타는 NULL
    )
    db.add(dst); db.commit(); db.refresh(dst)
    return dst


@router.post("/onnx", status_code=status.HTTP_202_ACCEPTED)
async def request_onnx_export(
    request: OnnxConversionRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    ONNX 모델 변환 요청
    
    PT 모델을 ONNX 포맷으로 변환합니다.
    
    **지원 Precision:**
    - fp32: Full precision (기본값)
    - fp16: Half precision (더 빠름, 정확도 약간 감소)
    - int8: Quantized (가장 빠름, 정확도 감소)
    
    **요청 예시:**
    ```json
    {
      "job_id": "optional",
      "model": {"s3_uri": "s3://visioninapp-bucket/models/xxx/best.pt"},
      "output": {"prefix": "exports/onnx/xxx", "model_name": "model.onnx"},
      "ops": {
          "dynamic": true,
          "simplify": true,
          "opset": 13,
          "imgsz": 640,
          "precision": "fp32"
      }
    }
    ```
    """
    try:
        src_key = _key_from_s3_uri(request.model.s3_uri)
        out_prefix = request.output.prefix
        out_name = request.output.model_name or "best.onnx"

        # ONNX 아티팩트 선점 INSERT (v0에 귀속)
        _ensure_child_artifact(
            db, src_key=src_key, out_prefix=out_prefix, out_name=out_name, out_format="onnx"
        )

        # MQ 라우팅 (Pydantic 모델을 dict로 변환)
        payload = request.model_dump()
        payload["job_id"] = _job_id(request.job_id)
        # 참고: GPU가 업로드할 대상 key는 out_prefix/model_name (이미 DB에 선점 저장됨)
        send_onnx_request(payload)
        return {"ok": True, "job_id": payload["job_id"], "routed": "onnx.start", "exchange": "jobs.cmd"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to publish ONNX request: {e}")


@router.post("/trt", status_code=status.HTTP_202_ACCEPTED)
async def request_trt_export(
    request: TrtConversionRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    TensorRT 모델 변환 요청
    
    PT 모델을 TensorRT 엔진으로 변환합니다.
    NVIDIA GPU에서 최적화된 성능을 제공합니다.
    
    **지원 Precision:**
    - fp32: Full precision
    - fp16: Half precision (권장, 기본값)
    - int8: Quantized (가장 빠름, calibration 필요)
    
    **요청 예시:**
    ```json
    {
      "job_id": "optional",
      "model": {"s3_uri": "s3://visioninapp-bucket/models/xxx/best.pt"},
      "output": {"prefix": "exports/trt/xxx", "model_name": "model.engine"},
      "trt": {
          "precision": "fp16",
          "imgsz": 640,
          "dynamic": true
      }
    }
    ```
    """
    try:
        src_key = _key_from_s3_uri(request.model.s3_uri)
        out_prefix = request.output.prefix
        out_name = request.output.model_name or "best.engine"

        # TRT 아티팩트 선점 INSERT (v0에 귀속)
        _ensure_child_artifact(
            db, src_key=src_key, out_prefix=out_prefix, out_name=out_name, out_format="engine"
        )

        # MQ 라우팅 (Pydantic 모델을 dict로 변환)
        payload = request.model_dump()
        payload["job_id"] = _job_id(request.job_id)
        send_trt_request(payload)
        return {"ok": True, "job_id": payload["job_id"], "routed": "trt.start", "exchange": "jobs.cmd"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to publish TRT request: {e}")