# graph/training/nodes/train_trial.py
from __future__ import annotations

import json
import os
import uuid
import time
from typing import Any, Dict

from graph.training.state import TrainState

# --- RabbitMQ settings ---
RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
EXCHANGE    = os.getenv("RMQ_EXCHANGE", "jobs.cmd")   # topic exchange
RK_START    = "train.start"                           # 학습 요청 발행
RK_DONE_FMT = "job.{job_id}.done"                     # GPU 서버 완료 알림 라우팅키
S3_BUCKET = os.getenv("S3_BUCKET", "visioninapp-bucket")

# ------------------------ 유틸 ------------------------

def _merge_train_params(state: TrainState) -> Dict[str, Any]:
    cfg = state.train_cfg or {}
    base = (cfg.get("train") or {}).copy()
    best = ((state.best_trial or {}).get("params") or {}).copy()
    over = (state.train_overrides or {}).copy()
    merged = {**base, **best, **over}
    merged.setdefault("epochs", 100)
    merged.setdefault("batch", 16)
    merged.setdefault("imgsz", 640)
    merged.setdefault("model", "yolo12n")
    return {k: v for k, v in merged.items() if v is not None}


def _infer_dataset(state: TrainState) -> Dict[str, str]:
    over = state.train_overrides or {}
    if isinstance(over.get("dataset"), dict):
        ds = over["dataset"]
        name = str(ds.get("name") or "").strip()
        s3_prefix = str(ds.get("s3_prefix") or "").strip()
        if name and s3_prefix:
            return {"name": name, "s3_prefix": s3_prefix}

    cfg = state.train_cfg or {}
    data_cfg = cfg.get("data") or {}
    ver = (state.dataset_version or data_cfg.get("dataset_version") or "").strip()
    name = ver.split("@")[0] if "@" in ver else (ver or "dataset").strip()
    if not name:
        name = "dataset"
    return {"name": name, "s3_prefix": f"datasets/{name}/"}


def _infer_output(state: TrainState, dataset_name: str) -> Dict[str, str]:
    over = state.train_overrides or {}
    if isinstance(over.get("output"), dict):
        out = over["output"]
        prefix = str(out.get("prefix") or "").strip()
        model_name = str(out.get("model_name") or "").strip()
        metrics_name = str(out.get("metrics_name") or "").strip() or "results.csv"
        if prefix and model_name:
            return {"s3_bucket": S3_BUCKET, "prefix": prefix, "model_name": model_name, "metrics_name": metrics_name}

    return {
        "prefix": f"models/{dataset_name}",
        "model_name": f"{dataset_name}.pt",
        "metrics_name": "results.csv",
    }


# ------------------- RabbitMQ 통신 -------------------

def _publish_to_rabbitmq(message: Dict[str, Any]) -> None:
    try:
        import pika
    except Exception as e:
        raise RuntimeError(f"pika 미설치 또는 로딩 실패: {e}") from e

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")

    ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=RK_START,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )
    conn.close()


def _wait_for_done(job_id: str, timeout_sec: int = 10800) -> Dict[str, Any]:
    """GPU 서버가 job.{job_id}.done 메시지를 보낼 때까지 대기"""
    import pika
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)

    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    qname = q.method.queue
    rk_done = RK_DONE_FMT.format(job_id=job_id)
    ch.queue_bind(exchange=EXCHANGE, queue=qname, routing_key=rk_done)

    deadline = time.monotonic() + timeout_sec
    result_payload = None

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

    ch.queue_unbind(exchange=EXCHANGE, queue=qname, routing_key=rk_done)
    conn.close()

    if result_payload is None:
        return {"job_id": job_id, "status": "timeout", "error": "no done event received"}
    return result_payload


# ------------------- 메인 노드 -------------------

def train_trial(state: TrainState) -> TrainState:
    """
    EC2 → GPU 학습 요청 발행 전용
    절대 학습 수행 금지. 메시지 구조는 GPU 서버 요구사항에 맞춤.
    """
    print("[train_trial] Starting train_trial node...")
    job_id = (state.job_id or str(uuid.uuid4())).replace(" ", "")
    print(f"[train_trial] Job ID: {job_id}")
    merged = _merge_train_params(state)
    ds = _infer_dataset(state)
    out = _infer_output(state, ds["name"])
    print(f"[train_trial] Dataset: {ds}, Output: {out}")

    # split 정보가 있으면 추가
    over = state.train_overrides or {}
    split = over.get("split")
    split_seed = over.get("split_seed")
    move_files = over.get("move_files")

    payload = {
        "job_id": job_id,
        "dataset": ds,
        "output": out,
        "hyperparams": merged,     # GPU가 학습 시 사용할 파라미터
    }

    # optional fields
    if split is not None:
        payload["split"] = split
    if split_seed is not None:
        payload["split_seed"] = split_seed
    if move_files is not None:
        payload["move_files"] = move_files

    # 1️⃣ 학습 요청 발행
    print(f"[train_trial] Publishing training request to RabbitMQ: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    try:
        _publish_to_rabbitmq(payload)
        print(f"[train_trial] ✅ Training request published successfully")
    except Exception as e:
        print(f"[train_trial] ❌ Failed to publish training request: {e}")
        state.error = f"Failed to publish training request: {str(e)}"
        state.action = "TRAIN_FAILED"
        return state

    # 2️⃣ 완료 이벤트 대기 (job.{job_id}.done)
    wait_sec = int(os.getenv("TRAIN_WAIT_TIMEOUT_SEC", "10800"))
    print(f"[train_trial] Waiting for GPU server response (timeout: {wait_sec}s)...")
    result = _wait_for_done(job_id, wait_sec)
    print(f"[train_trial] Received result: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # 3️⃣ 결과 반영
    ctx = state.context or {}
    ctx["train_trial"] = {
        "exchange": EXCHANGE,
        "rk_start": RK_START,
        "rk_done": RK_DONE_FMT.format(job_id=job_id),
        "payload": payload,
        "result": result,
    }
    state.context = ctx
    state.job_id = job_id

    # 4️⃣ 결과 상태 정리
    if result.get("event") == "done":
        artifact = result.get("artifact") or {}
        metrics = result.get("metrics") or {}
        state.model_path = artifact.get("model_path") or artifact.get("s3_path")
        state.metrics = metrics
        state.action = "TRAIN_COMPLETED"
        return state

    elif result.get("status") == "timeout":
        state.action = "TRAIN_TIMEOUT"
        state.error = result.get("error")
        return state

    else:
        state.action = "TRAIN_FAILED"
        state.error = result.get("error") or "unknown error"
        return state
