# graph/training/nodes/train_trial.py
from __future__ import annotations

import json
import os
import uuid
import time
from typing import Any, Dict
from tools.s3_client import download_s3

from graph.training.state import TrainState

# --- RabbitMQ settings ---
RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
# 학습 요청 보낼 exchange (GPU 서버 main.py에서 train 큐 바인딩된 cmd용)
EXCHANGE_CMD = os.getenv("RMQ_EXCHANGE_CMD", "jobs.cmd")

# 진행률/완료 이벤트 받을 exchange (Progress에서 사용하는 events용)
EXCHANGE_EVENTS = os.getenv("RMQ_EXCHANGE_EVENTS", "jobs.events")

RK_START = "train.start"             # 학습 요청
RK_DONE_FMT = "job.{job_id}.done"    # 완료 이벤트 routing key
S3_BUCKET = os.getenv("S3_BUCKET", "visioninapp-bucket")

# ------------------------ 유틸 ------------------------

def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("null", "none"):
            return None
        return s
    # str이 아니어도 들어오면 문자열로 캐스팅
    s = str(val).strip()
    return s or None


def _select_model_from_params(params: Dict[str, Any]) -> str | None:
    """
    모델 이름 선택 규칙:
    1) model
    2) model_name
    3) model_variant
    4) model_varient (오타 대응)
    위 순서대로 유효한 값을 찾는다.
    """
    for key in ("model", "model_name", "model_variant"):
        v = _clean_str(params.get(key))
        if v:
            return v
    return None

def _merge_train_params(state: TrainState) -> Dict[str, Any]:
    cfg = state.train_cfg or {}

    base = (cfg.get("train") or {}).copy()
    best = ((state.best_trial or {}).get("params") or {}).copy()
    over = (state.train_overrides or {}).copy()

    # 먼저 base/best/over를 한 데 합치고
    merged = {**base, **best, **over}

    # 🔹 여기서 모델 이름을 안정적으로 뽑는다
    selected_model = _select_model_from_params(merged)
    if selected_model:
        merged["model"] = selected_model
    else:
        # 유효한 값이 진짜로 하나도 없을 때만 기본값 사용
        merged["model"] = "yolo12n"

    # 기본값들 (이미 값 있으면 건들지 않음)
    merged.setdefault("epochs", 100)
    merged.setdefault("batch", 16)
    merged.setdefault("imgsz", 640)

    # None / "null" 등은 깔끔하게 제거
    cleaned = {}
    for k, v in merged.items():
        if isinstance(v, str):
            vv = v.strip()
            if not vv or vv.lower() in ("null", "none"):
                continue
            cleaned[k] = vv
        elif v is not None:
            cleaned[k] = v

    return cleaned


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
        "prefix": f"models/{dataset_name}/train",
        "model_name": f"{dataset_name}.pt",
        "metrics_name": "results.csv",
    }


# ------------------- RabbitMQ 통신 -------------------

def _publish_to_rabbitmq(message: Dict[str, Any]) -> None:
    import pika

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    # 요청은 cmd exchange로
    ch.exchange_declare(exchange=EXCHANGE_CMD, exchange_type="topic", durable=True)

    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    ch.basic_publish(
        exchange=EXCHANGE_CMD,
        routing_key=RK_START,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )
    conn.close()



def _wait_for_done(job_id: str, timeout_sec: int = 10800) -> Dict[str, Any]:
    """
    GPU 서버가 events exchange (EXCHANGE_EVENTS)에
    job.{job_id}.done 메시지를 보낼 때까지 대기
    """
    import pika

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    # ✅ done/progress 는 events exchange 에서 온다
    ch.exchange_declare(exchange=EXCHANGE_EVENTS, exchange_type="topic", durable=True)

    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    qname = q.method.queue

    rk_done = RK_DONE_FMT.format(job_id=job_id)
    ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)

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

        # Progress.done 구조와 일치하는지 확인
        if (
            str(data.get("job_id")) == job_id
            and data.get("event") == "done"
        ):
            result_payload = data
            ch.basic_ack(method.delivery_tag)
            break

        ch.basic_ack(method.delivery_tag)

    ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)
    conn.close()

    if result_payload is None:
        return {
            "job_id": job_id,
            "status": "timeout",
            "error": "no done event received",
        }
    return result_payload



# ------------------- 메인 노드 -------------------

def train_trial(state: TrainState) -> TrainState:
    """
    EC2 → GPU 학습 요청 발행 전용
    절대 학습 수행 금지. 메시지 구조는 GPU 서버 요구사항에 맞춤.
    """
    job_id = (state.job_id or str(uuid.uuid4())).replace(" ", "")
    merged = _merge_train_params(state)
    ds = _infer_dataset(state)
    out = _infer_output(state, ds["name"])

    # split 정보가 있으면 추가
    over = state.train_overrides or {}

    if "model" not in over:
        if "model_name" in over:
            over["model"] = over["model_name"]
        elif "model_variant" in over:
            over["model"] = over["model_variant"]

    state.train_overrides = over

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
    print(payload)
    # 1️⃣ 학습 요청 발행
    _publish_to_rabbitmq(payload)

    # 2️⃣ 완료 이벤트 대기 (job.{job_id}.done)
    wait_sec = int(os.getenv("TRAIN_WAIT_TIMEOUT_SEC", "10800"))
    result = _wait_for_done(job_id, wait_sec)

    # 3️⃣ 결과 반영
    ctx = state.context or {}
    ctx["train_trial"] = {
        "exchange": EXCHANGE_CMD,
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
