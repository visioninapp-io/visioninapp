from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.core.database import get_db
from app.core.auth import get_current_user
from app.rabbitmq.producer import send_onnx_request, send_trt_request

router = APIRouter()

def _job_id(s: Optional[str] = None) -> str:
    return s if s else uuid.uuid4().hex[:8]

@router.post("/onnx", status_code=status.HTTP_202_ACCEPTED)
async def request_onnx_export(
    payload: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    요청 예시:
    {
      "job_id": "optional",
      "model": {"s3_uri": "s3://visioninapp-bucket/models/xxx/best.pt"},
      "output": {"prefix": "exports/onnx/xxx", "model_name": "best.onnx"},
      "ops": {"dynamic": true, "simplify": true}
    }
    """
    try:
        if "model" not in payload or "s3_uri" not in payload["model"]:
            raise HTTPException(400, "model.s3_uri is required")
        if "output" not in payload or "prefix" not in payload["output"]:
            raise HTTPException(400, "output.prefix is required")

        payload["job_id"] = _job_id(payload.get("job_id"))
        send_onnx_request(payload)
        return {"ok": True, "job_id": payload["job_id"], "routed": "onnx.start", "exchange": "jobs.cmd"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to publish ONNX request: {e}")

@router.post("/trt", status_code=status.HTTP_202_ACCEPTED)
async def request_trt_export(
    payload: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    요청 예시:
    {
      "job_id": "optional",
      "model": {"s3_uri": "s3://visioninapp-bucket/models/xxx/best.pt"},
      "trt": {"precision": "fp16", "imgsz": 640, "dynamic": true},
      "output": {"prefix": "exports/trt/xxx", "model_name": "best.engine"}
    }
    """
    try:
        if "model" not in payload or "s3_uri" not in payload["model"]:
            raise HTTPException(400, "model.s3_uri is required")
        if "output" not in payload or "prefix" not in payload["output"]:
            raise HTTPException(400, "output.prefix is required")

        payload["job_id"] = _job_id(payload.get("job_id"))
        send_trt_request(payload)
        return {"ok": True, "job_id": payload["job_id"], "routed": "trt.start", "exchange": "jobs.cmd"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to publish TRT request: {e}")