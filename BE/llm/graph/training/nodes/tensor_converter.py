from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm.graph.training.state import TrainState

# --- RabbitMQ settings ---
RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F",
)

EXCHANGE_CMD = os.getenv("RMQ_EXCHANGE_CMD", os.getenv("RMQ_EXCHANGE", "jobs.cmd"))
EXCHANGE_EVENTS = os.getenv("RMQ_EXCHANGE_EVENTS", "jobs.events")

# ✅ TensorRT 작업 라우팅키: trt.start 큐와 바인딩
RK_TRT = os.getenv("RMQ_RK_TRT", "trt.start")

RK_DONE_FMT = "job.{job_id}.done"

S3_BUCKET_DEFAULT = os.getenv("S3_BUCKET", "visioninapp-bucket")


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"invalid s3 uri: {uri}")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid s3 uri: {uri}")
    return parts[0], parts[1]


def _infer_model_s3_uri(state: TrainState) -> str:
    ctx = getattr(state, "context", {}) or {}
    tt = ctx.get("train_trial") or {}
    res = (tt.get("result") or {})
    art = (res.get("artifact") or {})
    uri = art.get("s3_uri")
    if uri:
        return uri

    mp = getattr(state, "model_path", None)
    if isinstance(mp, str) and mp.startswith("s3://"):
        return mp

    raise RuntimeError(
        "TensorRT 변환용 S3 모델 경로를 찾지 못했습니다. "
        "train_trial 결과에 artifact.s3_uri가 설정되어 있어야 합니다."
    )


def _get_trt_options(state: TrainState) -> Dict[str, Any]:
    over = state.train_overrides or {}
    trt = over.get("trt") or {}

    precision = (trt.get("precision") or "fp16").lower()
    imgsz = int(trt.get("imgsz") or over.get("imgsz") or 640)
    if imgsz <= 0:
        imgsz = 640
    dynamic = bool(trt.get("dynamic")) if "dynamic" in trt else True

    return {
        "precision": precision,
        "imgsz": imgsz,
        "dynamic": dynamic,
    }


def _build_trt_output(s3_uri: str, job_id: str) -> Dict[str, Any]:
    bucket, key = _split_s3_uri(s3_uri)
    base_dir = os.path.dirname(key)
    prefix = f"{base_dir}/trt/"
    return {
        "s3_bucket": bucket or S3_BUCKET_DEFAULT,
        "prefix": prefix,
        "model_name": "best.engine",
    }


def _publish_cmd(message: Dict[str, Any], routing_key: str) -> None:
    import pika

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    ch.exchange_declare(exchange=EXCHANGE_CMD, exchange_type="topic", durable=True)

    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    ch.basic_publish(
        exchange=EXCHANGE_CMD,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )
    conn.close()


def _wait_for_done(job_id: str, timeout_sec: int) -> Dict[str, Any]:
    import pika

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    ex_events = EXCHANGE_EVENTS or EXCHANGE_CMD
    ch.exchange_declare(exchange=ex_events, exchange_type="topic", durable=True)

    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    qname = q.method.queue

    rk_done = RK_DONE_FMT.format(job_id=job_id)
    ch.queue_bind(exchange=ex_events, queue=qname, routing_key=rk_done)

    deadline = time.monotonic() + timeout_sec
    result_payload: Optional[Dict[str, Any]] = None

    for method, properties, body in ch.consume(qname, inactivity_timeout=1.0):
        if method is None:
            if time.monotonic() > deadline:
                break
            continue

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {"status": "error", "error": "invalid JSON"}

        if str(data.get("job_id")) == job_id and data.get("event") == "done":
            result_payload = data
            ch.basic_ack(method.delivery_tag)
            break

        ch.basic_ack(method.delivery_tag)

    ch.queue_unbind(exchange=ex_events, queue=qname, routing_key=rk_done)
    conn.close()

    if result_payload is None:
        return {
            "job_id": job_id,
            "status": "timeout",
            "error": "no done event received",
        }
    return result_payload


def tensor_converter(state: TrainState) -> TrainState:
    """
    백엔드 전용:
    - GPU 서버(main.py handle_trt)가 TensorRT 엔진 생성.
    - 여기서는 YOLO/TensorRT 직접 실행 금지.
    """
    print("[tensor_converter] request TensorRT convert via GPU server")

    base_job = (state.job_id or str(uuid.uuid4())).replace(" ", "")
    job_id = f"{base_job}.trt"

    model_s3_uri = _infer_model_s3_uri(state)
    trt_opts = _get_trt_options(state)
    output_cfg = _build_trt_output(model_s3_uri, job_id)

    msg = {
        "job_id": job_id,
        "model": {"s3_uri": model_s3_uri},
        "trt": trt_opts,
        "output": output_cfg,
    }

    _publish_cmd(msg, RK_TRT)

    wait_sec = int(os.getenv("TRT_WAIT_TIMEOUT_SEC", "3600"))
    result = _wait_for_done(job_id, wait_sec)

    ctx = state.context or {}
    ctx["tensor_converter"] = {"job_id": job_id, "request": msg, "result": result}
    state.context = ctx

    if result.get("event") == "done":
        artifact = result.get("artifact") or {}
        trt_s3 = artifact.get("s3_uri") or artifact.get("engine_s3_uri")

        reg = dict(state.registry_info or {})

        # ✅ 마찬가지로 path가 없다면 세팅
        if "path" not in reg:
            model_s3_uri = _infer_model_s3_uri(state)
            bucket, key = _split_s3_uri(model_s3_uri)
            base_dir = os.path.dirname(key)
            reg["bucket"] = bucket
            reg["path"] = f"s3://{bucket}/{base_dir}"

        if trt_s3:
            reg["engine_s3_uri"] = trt_s3

        state.registry_info = reg
        state.action = "TRT_CONVERTED"
        print("[tensor_converter] exchange complete!")
        return state
