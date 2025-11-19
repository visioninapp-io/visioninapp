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
RK_ERROR_FMT = "job.{job_id}.error"
RK_STATUS_FMT = "job.{job_id}.status"

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


def _wait_for_done(job_id: str, timeout_sec: int = 21600) -> Dict[str, Any]:
    """
    GPU 서버에서 전송하는 job.{job_id}.done / job.{job_id}.error / job.{job_id}.status 이벤트를 처리.
    - done: 성공으로 종료
    - error: 즉시 실패로 종료
    - status/progress: 진행률 로그만 남기고 계속 대기
    """
    import pika

    try:
        print(f"[train_trial] 완료 이벤트 대기 시작: job_id={job_id}, timeout={timeout_sec}초")
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()

        print(f"[train_trial] Exchange 선언: {EXCHANGE_EVENTS}")
        ch.exchange_declare(exchange=EXCHANGE_EVENTS, exchange_type="topic", durable=True)

        q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
        qname = q.method.queue

        rk_done = RK_DONE_FMT.format(job_id=job_id)
        rk_error = RK_ERROR_FMT.format(job_id=job_id)
        rk_status = RK_STATUS_FMT.format(job_id=job_id)

        # ✅ 성공 / 실패 / 진행률 모두 구독
        print(f"[train_trial] 큐 바인딩: {qname} <- {rk_done}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)
        print(f"[train_trial] 큐 바인딩: {qname} <- {rk_error}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_error)
        print(f"[train_trial] 큐 바인딩: {qname} <- {rk_status}")
        ch.queue_bind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_status)

        deadline = time.monotonic() + timeout_sec
        result_payload: Dict[str, Any] | None = None
        last_log = time.monotonic()

        for method, properties, body in ch.consume(qname, inactivity_timeout=1.0):
            now = time.monotonic()

            # 주기 로그
            if now - last_log >= 10:
                remain = max(0, int(deadline - now))
                print(f"[train_trial] 대기 중... (남은 시간: {remain}초)")
                last_log = now

            # inactivity_timeout: 메시지 없음
            if method is None:
                if now > deadline:
                    print(f"[train_trial] ⏰ 타임아웃: {timeout_sec}초 경과")
                    break
                continue

            raw = body.decode("utf-8", errors="replace")

            try:
                data = json.loads(raw)
            except Exception as e:
                print(f"[train_trial] ⚠️ JSON 파싱 오류: {e}, body={raw!r}")
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

            print(f"[train_trial] 이벤트 수신: rk={rk}, event={event}, data={data}")

            # 🔹 다른 job_id면 무시
            if msg_job_id and msg_job_id != job_id:
                ch.basic_ack(method.delivery_tag)
                continue

            # 🔹 진행률 이벤트: status/progress
            if rk == rk_status or event in ("progress", "status"):
                epoch = data.get("epoch")
                total = data.get("total_epochs")
                if epoch is not None and total is not None:
                    print(f"[train_trial] 진행률: {epoch}/{total} epoch 완료")
                else:
                    print(f"[train_trial] 진행률 이벤트 수신: {data}")
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
                print(f"[train_trial] ⏰ 타임아웃: {timeout_sec}초 경과")
                break

        # 언바인딩 및 연결 종료
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_done)
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_error)
        ch.queue_unbind(exchange=EXCHANGE_EVENTS, queue=qname, routing_key=rk_status)
        conn.close()

        # 아무 결과도 못 받음 → 타임아웃 처리
        if result_payload is None:
            print(f"[train_trial] ⚠️ 완료/에러 이벤트 미수신")
            return {
                "job_id": job_id,
                "event": "timeout",
                "status": "timeout",
                "error": "no done/error event received",
            }

        return result_payload

    except Exception as e:
        print(f"[train_trial] ❌ 완료 이벤트 대기 중 예외 발생: {e}")
        return {
            "job_id": job_id,
            "event": "error",
            "status": "error",
            "error": f"_wait_for_done exception: {e}",
        }


def tensor_converter(state: TrainState) -> TrainState:
    """
    백엔드 전용:
    - GPU 서버(main.py handle_trt)가 TensorRT 엔진 생성.
    - 여기서는 YOLO/TensorRT 직접 실행 금지.
    """
    print("[tensor_converter] request TensorRT convert via GPU server")

    base_job = (state.job_id or str(uuid.uuid4()).replace("-", ""))
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

    # 프론트엔드에 변환 정보 알림 (convert.exchanges)
    try:
        import pika
        import json
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.exchange_declare(exchange="jobs.event", exchange_type="topic", durable=True)
        
        conversion_info = {
            "job_id": base_job,  # 원본 job_id 사용 (프론트엔드가 추적 중인 job_id)
            "onnx": "false",
            "tensorrt": "true"
        }
        body = json.dumps(conversion_info, ensure_ascii=False).encode("utf-8")
        ch.basic_publish(
            exchange="jobs.event",
            routing_key="convert.exchanges",
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        conn.close()
        print(f"[tensor_converter] Sent conversion info to frontend: {conversion_info}")
    except Exception as e:
        print(f"[tensor_converter] Failed to send conversion info: {e}")

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
