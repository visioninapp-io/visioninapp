# graph/training/nodes/train_trial.py
from __future__ import annotations

import json
import os
import uuid
import time
from typing import Any, Dict
from llm.tools.s3_client import download_s3

from llm.graph.training.state import TrainState
import logging

logger = logging.getLogger("uvicorn.error")

# --- RabbitMQ settings ---
RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
# 학습 요청 보낼 exchange (GPU 서버 main.py에서 train 큐 바인딩된 cmd용)
EXCHANGE_CMD = os.getenv("RMQ_EXCHANGE_CMD", "jobs.cmd")

# 진행률/완료 이벤트 받을 exchange (Progress에서 사용하는 events용)
EXCHANGE_EVENTS = os.getenv("RMQ_EXCHANGE_EVENTS", "jobs.events")

RK_START = "train.start"             # 학습 요청
RK_HPO = "train.hpo"
RK_DONE_FMT = "job.{job_id}.done"    # 완료 이벤트 routing key
RK_ERROR_FMT = "job.{job_id}.error"
RK_STATUS_FMT = "job.{job_id}.status"
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

def _publish_to_rabbitmq(message: Dict[str, Any], rk: str) -> None:
    import pika

    try:
        logger.info(f"[train_trial] RabbitMQ 연결 시도: {RABBITMQ_URL}")
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()

        # 요청은 cmd exchange로
        logger.info(f"[train_trial] Exchange 선언: {EXCHANGE_CMD}")
        ch.exchange_declare(exchange=EXCHANGE_CMD, exchange_type="topic", durable=True)

        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        logger.info(f"[train_trial] 메시지 발행: exchange={EXCHANGE_CMD}, routing_key={RK_START}")
        ch.basic_publish(
            exchange=EXCHANGE_CMD,
            routing_key=rk,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        conn.close()
        logger.info(f"[train_trial] ✅ 메시지 발행 완료")
    except Exception as e:
        logger.info(f"[train_trial] ❌ RabbitMQ 메시지 발행 실패: {e}")
        raise



def _wait_for_done(job_id: str, timeout_sec: int = 21600) -> Dict[str, Any]:
    """
    GPU 서버에서 전송하는 job.{job_id}.done / job.{job_id}.error / job.{job_id}.status 이벤트를 처리.
    - done: 성공으로 종료
    - error: 즉시 실패로 종료
    - status/progress: 진행률 로그만 남기고 계속 대기
    """
    import pika

    try:
        logger.info(f"[train_trial] 완료 이벤트 대기 시작: job_id={job_id}, timeout={timeout_sec}초")
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()

        logger.info(f"[train_trial] Exchange 선언: {EXCHANGE_EVENTS}")
        ch.exchange_declare(exchange=EXCHANGE_EVENTS, exchange_type="topic", durable=True)

        q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
        qname = q.method.queue

        rk_done = RK_DONE_FMT.format(job_id=job_id)
        rk_error = RK_ERROR_FMT.format(job_id=job_id)
        rk_status = RK_STATUS_FMT.format(job_id=job_id)

        # ✅ 성공 / 실패 / 진행률 모두 구독
        logger.info(f"[train_trial] 큐 바인딩: {qname} <- {rk_done}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)
        logger.info(f"[train_trial] 큐 바인딩: {qname} <- {rk_error}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_error)
        logger.info(f"[train_trial] 큐 바인딩: {qname} <- {rk_status}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_status)

        deadline = time.monotonic() + timeout_sec
        result_payload: Dict[str, Any] | None = None
        last_log = time.monotonic()

        for method, properties, body in ch.consume(qname, inactivity_timeout=1.0):
            now = time.monotonic()

            # 주기 로그
            if now - last_log >= 10:
                remain = max(0, int(deadline - now))
                logger.info(f"[train_trial] 대기 중... (남은 시간: {remain}초)")
                last_log = now

            # inactivity_timeout: 메시지 없음
            if method is None:
                if now > deadline:
                    logger.info(f"[train_trial] ⏰ 타임아웃: {timeout_sec}초 경과")
                    break
                continue

            raw = body.decode("utf-8", errors="replace")

            try:
                data = json.loads(raw)
            except Exception as e:
                logger.info(f"[train_trial] ⚠️ JSON 파싱 오류: {e}, body={raw!r}")
                # JSON 깨졌으면 이 job은 실패 처리
                result_payload = {
                    "job_id": job_id,
                    "event": "error",
                    "status": "error",
                    "error": f"invalid JSON from GPU server: {e}",
                }
                ch.basic_ack(method.delivery_tag)
                break

            msg_job_id = str(data.get("job_id") or "")
            event = (data.get("event") or data.get("status") or "").lower()
            rk = method.routing_key

            logger.info(f"[train_trial] 이벤트 수신: rk={rk}, event={event}, data={data}")

            # 🔹 다른 job_id면 무시
            if msg_job_id and msg_job_id != job_id:
                ch.basic_ack(method.delivery_tag)
                continue

            # 🔹 진행률 이벤트: status/progress
            if rk == rk_status or event in ("progress", "status"):
                epoch = data.get("epoch")
                total = data.get("total_epochs")
                if epoch is not None and total is not None:
                    logger.info(f"[train_trial] 진행률: {epoch}/{total} epoch 완료")
                else:
                    logger.info(f"[train_trial] 진행률 이벤트 수신: {data}")
                ch.basic_ack(method.delivery_tag)
                continue  # 계속 다음 메시지 대기

            # 🔹 실패 이벤트: error routing key 또는 event == error/failed
            if rk == rk_error or event in ("error", "failed", "train_error"):
                result_payload = {
                    **data,
                    "job_id": msg_job_id or job_id,
                    "event": "error",
                    "status": "error",
                }
                ch.basic_ack(method.delivery_tag)
                break

            # 🔹 성공 이벤트: done + event == done
            if rk == rk_done and event == "done":
                status = (data.get("status") or "success").lower()
                result_payload = {
                    **data,
                    "job_id": msg_job_id or job_id,
                    "event": "done",
                    "status": status,
                }
                ch.basic_ack(method.delivery_tag)
                break

            # 🔹 그 외는 소비만 하고 무시
            ch.basic_ack(method.delivery_tag)

            if now > deadline:
                logger.info(f"[train_trial] ⏰ 타임아웃: {timeout_sec}초 경과")
                break

        # 언바인딩 및 연결 종료
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_error)
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_status)
        conn.close()

        # 아무 결과도 못 받음 → 타임아웃 처리
        if result_payload is None:
            logger.info(f"[train_trial] ⚠️ 완료/에러 이벤트 미수신")
            return {
                "job_id": job_id,
                "event": "timeout",
                "status": "timeout",
                "error": "no done/error event received",
            }

        return result_payload

    except Exception as e:
        logger.info(f"[train_trial] ❌ 완료 이벤트 대기 중 예외 발생: {e}")
        return {
            "job_id": job_id,
            "event": "error",
            "status": "error",
            "error": f"_wait_for_done exception: {e}",
        }


# ------------------- 메인 노드 -------------------

def train_trial(state: TrainState) -> TrainState:
    """
    EC2 → GPU 학습 요청 발행 전용
    절대 학습 수행 금지. 메시지 구조는 GPU 서버 요구사항에 맞춤.
    """
    job_id = state.job_id or str(uuid.uuid4()).replace("-", "")
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

    hpo = {
        "job_id": job_id,
        "hyperparams": merged,
    }

    # optional fields
    if split is not None:
        payload["split"] = split
    if split_seed is not None:
        payload["split_seed"] = split_seed
    if move_files is not None:
        payload["move_files"] = move_files
    logger.info(payload)
    # 1️⃣ 학습 요청 발행
    logger.info(f"[train_trial] 학습 요청 발행 시작: job_id={job_id}")
    _publish_to_rabbitmq(payload, RK_START)
    _publish_to_rabbitmq(hpo, RK_HPO)

    # 2️⃣ 완료 이벤트 대기 (job.{job_id}.done)
    wait_sec = int(os.getenv("TRAIN_WAIT_TIMEOUT_SEC", "10800"))
    result = _wait_for_done(job_id, wait_sec)
    logger.info(f"[train_trial] 완료 이벤트 대기 종료: result={result.get('status', 'unknown')}")

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
    event = result.get("event")
    status = result.get("status")

    if event == "done" and status not in ("error", "failed"):
        artifact = result.get("artifact") or {}
        metrics = result.get("metrics") or {}
        state.model_path = artifact.get("model_path") or artifact.get("s3_path")
        state.metrics = metrics
        state.action = "TRAIN_COMPLETED"
        return state

    if status == "timeout":
        state.action = "TRAIN_TIMEOUT"
        state.error = result.get("error") or "no done/error event received"
        return state

    # 그 외는 전부 실패로 처리 (event == "error" 포함)
    state.action = "TRAIN_FAILED"
    state.error = result.get("error") or f"train failed: {status or event}"
    return state
