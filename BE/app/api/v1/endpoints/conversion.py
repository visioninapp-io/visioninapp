from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.config import settings
from app.models.model_artifact import ModelArtifact
from app.models.model_version import ModelVersion
from app.rabbitmq.producer import send_onnx_request, send_trt_request

router = APIRouter()

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
    payload: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    요청 예시(형식 변경 없음):
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

        src_key = _key_from_s3_uri(payload["model"]["s3_uri"])
        out_prefix = payload["output"]["prefix"]
        out_name = payload["output"].get("model_name", "best.onnx")

        # ONNX 아티팩트 선점 INSERT (v0에 귀속)
        _ensure_child_artifact(
            db, src_key=src_key, out_prefix=out_prefix, out_name=out_name, out_format="onnx"
        )

        # MQ 라우팅 (바디 포맷 그대로)
        payload["job_id"] = _job_id(payload.get("job_id"))
        # 참고: GPU가 업로드할 대상 key는 out_prefix/model_name (이미 DB에 선점 저장됨)
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
    요청 예시(형식 변경 없음):
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

        src_key = _key_from_s3_uri(payload["model"]["s3_uri"])
        out_prefix = payload["output"]["prefix"]
        out_name = payload["output"].get("model_name", "best.engine")

        # TRT 아티팩트 선점 INSERT (v0에 귀속)
        _ensure_child_artifact(
            db, src_key=src_key, out_prefix=out_prefix, out_name=out_name, out_format="engine"
        )

        payload["job_id"] = _job_id(payload.get("job_id"))
        send_trt_request(payload)
        return {"ok": True, "job_id": payload["job_id"], "routed": "trt.start", "exchange": "jobs.cmd"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to publish TRT request: {e}")