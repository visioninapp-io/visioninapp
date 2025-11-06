# graph/training/nodes/train_trial.py
from __future__ import annotations

import json
import os
import uuid
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from graph.training.state import TrainState

load_dotenv()
# --- RabbitMQ settings ---
AMQP_URL    = os.getenv("RABBITMQ_URL", "")
EXCHANGE    = os.getenv("RMQ_EXCHANGE", "jobs.cmd")   # topic exchange
RK_START    = "train.start"                           # 발행
RK_DONE_FMT = "train.done.{job_id}"                   # 완료 수신 (GPU 서버가 여기에 publish)


def _merge_train_params(state: TrainState) -> Dict[str, Any]:
    """
    YAML.train 기본값 <- HPO best_trial.params <- 사용자 train_overrides
    """
    cfg = state.train_cfg or {}
    base = (cfg.get("train") or {}).copy()
    best = ((state.best_trial or {}).get("params") or {}).copy()
    over = (state.train_overrides or {}).copy()
    merged = {**base, **best, **over}
    merged.setdefault("epochs", 100)
    merged.setdefault("batch", 16)
    merged.setdefault("imgsz", 640)
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
    data_cfg = (cfg.get("data") or {})
    ver = (state.dataset_version or data_cfg.get("dataset_version") or "").strip()
    name = ver.split("@")[0] if "@" in ver else (ver or "dataset").strip() or "dataset"
    return {"name": name, "s3_prefix": f"datasets/{name}/"}


def _infer_output(state: TrainState, dataset_name: str) -> Dict[str, str]:
    over = state.train_overrides or {}
    if isinstance(over.get("output"), dict):
        out = over["output"]
        prefix = str(out.get("prefix") or "").strip()
        model_name = str(out.get("model_name") or "").strip()
        if prefix and model_name:
            return {"prefix": prefix, "model_name": model_name}
    return {"prefix": f"models/{dataset_name}", "model_name": f"{dataset_name}.pt"}


def _publish_to_rabbitmq(message: Dict[str, Any]) -> None:
    try:
        import pika
    except Exception as e:
        raise RuntimeError(f"pika 미설치 또는 로딩 실패: {e}") from e

    params = pika.URLParameters(AMQP_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=RK_START,
        body=body,
        properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
    )
    conn.close()


def _wait_for_completion(job_id: str, timeout_sec: int = 3 * 60 * 60) -> Dict[str, Any]:
    """
    train.done.{job_id} 라우팅키로 오는 완료 메시지를 기다린다.
    GPU 서버가 같은 EXCHANGE에 publish 해야 한다.
    """
    import pika

    params = pika.URLParameters(AMQP_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)

    # 임시 전용 큐를 만들고 이 job_id의 done 키에 바인딩
    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    qname = q.method.queue
    done_rk = RK_DONE_FMT.format(job_id=job_id)
    ch.queue_bind(exchange=EXCHANGE, queue=qname, routing_key=done_rk)

    deadline = time.monotonic() + timeout_sec
    result_payload: Optional[Dict[str, Any]] = None

    for method, properties, body in ch.consume(qname, inactivity_timeout=1.0):
        if method is None:
            # 타임슬라이스마다 타임아웃 확인
            if time.monotonic() > deadline:
                break
            continue
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {"status": "error", "error": "invalid json in done message"}
        # 해당 job_id인지 최종 확인 (방어)
        if str(data.get("job_id")) == job_id:
            result_payload = data
            ch.basic_ack(method.delivery_tag)
            break
        ch.basic_ack(method.delivery_tag)

    # 정리
    try:
        ch.queue_unbind(exchange=EXCHANGE, queue=qname, routing_key=done_rk)
    except Exception:
        pass
    conn.close()

    if result_payload is None:
        return {"job_id": job_id, "status": "timeout", "error": "training result timeout"}

    return result_payload


def train_trial(state: TrainState) -> TrainState:
    """
    [동기 대기 버전]
    - train.start 발행
    - train.done.{job_id} 수신까지 대기(타임아웃 포함)
    - 성공 시 metrics/model_path를 state에 채워 넣고 다음 단계로 진행
    """
    job_id = (state.job_id or str(uuid.uuid4())).replace(" ", "")

    hyper = _merge_train_params(state)
    ds = _infer_dataset(state)
    out = _infer_output(state, ds["name"])

    payload = {
        "job_id": job_id,
        "dataset": {"s3_prefix": ds["s3_prefix"], "name": ds["name"]},
        "output": {"prefix": out["prefix"], "model_name": out["model_name"]},
        "hyperparams": hyper,
    }

    # 1) 시작 발행
    _publish_to_rabbitmq(payload)

    # 2) 완료 대기 (기본 3시간)
    wait_sec = int(os.getenv("TRAIN_WAIT_TIMEOUT_SEC", "10800"))
    result = _wait_for_completion(job_id, timeout_sec=wait_sec)

    # 3) 결과 반영
    ctx = state.context or {}
    ctx["train_trial"] = {
        "dispatched": True,
        "exchange": EXCHANGE,
        "routing_key_start": RK_START,
        "routing_key_done": RK_DONE_FMT.format(job_id=job_id),
        "amqp_url": AMQP_URL,
        "request": payload,
        "result": result,
    }
    state.context = ctx
    state.job_id = job_id

    status = str(result.get("status", "")).lower()
    if status == "success":
        # GPU서버가 내려주는 스키마 예시:
        # {
        #   "job_id": "...",
        #   "status": "success",
        #   "output": {"s3_path": "s3://.../mymodel.pt", "local_path": "...", "model_name": "..."},
        #   "metrics": {"map50": 0.71, "map50-95": 0.42, ...},
        #   "logs_url": "..."
        # }
        out_info = result.get("output", {}) or {}
        state.model_path = out_info.get("s3_path") or out_info.get("local_path")
        state.metrics = result.get("metrics") or {}
        state.action = "TRAIN_COMPLETED"
        return state

    elif status == "timeout":
        state.error = result.get("error") or "training result timeout"
        state.action = "TRAIN_TIMEOUT"
        return state

    else:
        state.error = result.get("error") or "training failed"
        state.metrics = result.get("metrics") or {}
        state.action = "TRAIN_FAILED"
        return state
