import json, logging
from pika import BasicProperties
from app.rabbitmq.connection import get_channel

log = logging.getLogger(__name__)

EXCHANGE_CMD = "jobs.cmd"
RK_TRAIN_START = "train.start"
RK_ONNX_START  = "onnx.start"
RK_TRT_START   = "trt.start"
RK_INFERENCE_START = 'inference.start'

def _publish_with_job_binding(exchange: str, routing_key: str, payload: dict, job_id: str) -> None:
    """
    job_id 기반으로 동적 exchange, queue 선언 및 바인딩 후 publish, 그리고 즉시 해제

    Args:
        exchange: Exchange name (e.g., 'jobs.cmd')
        routing_key: Routing key (e.g., 'train.start')
        payload: Message payload (dict)
        job_id: Job ID for queue naming
    """
    conn, ch = get_channel()
    queue_name = f"job.{job_id}.temp.queue"

    try:
        # 1. Exchange 선언 (이미 존재하면 무시됨)
        ch.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
        log.info(f"[RMQ] Declared exchange: {exchange}")

        # 2. Queue 선언 (임시)
        ch.queue_declare(queue=queue_name, durable=False, auto_delete=True)
        log.info(f"[RMQ] Declared temp queue: {queue_name}")

        # 3. Exchange와 Queue를 routing_key로 바인딩
        ch.queue_bind(queue=queue_name, exchange=exchange, routing_key=routing_key)
        log.info(f"[RMQ] Bound queue '{queue_name}' to exchange '{exchange}' with routing_key '{routing_key}'")

        # 4. Publish 메시지
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        props = BasicProperties(content_type="application/json", delivery_mode=2)
        ch.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=props,
            mandatory=True,
        )
        log.info(f"[RMQ] Published message to {exchange}/{routing_key} job_id={job_id}")

    finally:
        # 5. 바인딩 해제 및 Queue 삭제
        try:
            ch.queue_unbind(queue=queue_name, exchange=exchange, routing_key=routing_key)
            log.info(f"[RMQ] Unbound queue '{queue_name}' from exchange '{exchange}'")
        except Exception as e:
            log.warning(f"[RMQ] Failed to unbind queue {queue_name}: {e}")

        try:
            ch.queue_delete(queue=queue_name)
            log.info(f"[RMQ] Deleted temp queue: {queue_name}")
        except Exception as e:
            log.warning(f"[RMQ] Failed to delete queue {queue_name}: {e}")

        try:
            ch.close()
        finally:
            conn.close()

def _publish(exchange: str, routing_key: str, payload: dict) -> None:
    """
    기본 publish 함수 (job_id가 없는 경우)
    """
    conn, ch = get_channel()
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        props = BasicProperties(content_type="application/json", delivery_mode=2)
        ch.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=props,
            mandatory=True,
        )
        log.info(f"[RMQ] published rk={routing_key} job_id={payload.get('job_id')}")
    finally:
        try: ch.close()
        finally: conn.close()

def send_train_request(payload: dict) -> None:
    """
    Train 요청 - job_id 기반 동적 바인딩 사용
    """
    job_id = payload.get('job_id')
    if job_id:
        _publish_with_job_binding(EXCHANGE_CMD, RK_TRAIN_START, payload, job_id)
    else:
        # job_id가 없으면 기본 publish (하위 호환성)
        _publish(EXCHANGE_CMD, RK_TRAIN_START, payload)

def send_onnx_request(payload: dict) -> None:
    """
    ONNX 변환 요청 - job_id 기반 동적 바인딩 사용
    """
    job_id = payload.get('job_id')
    if job_id:
        _publish_with_job_binding(EXCHANGE_CMD, RK_ONNX_START, payload, job_id)
    else:
        _publish(EXCHANGE_CMD, RK_ONNX_START, payload)

def send_trt_request(payload: dict) -> None:
    """
    TensorRT 변환 요청 - job_id 기반 동적 바인딩 사용
    """
    job_id = payload.get('job_id')
    if job_id:
        _publish_with_job_binding(EXCHANGE_CMD, RK_TRT_START, payload, job_id)
    else:
        _publish(EXCHANGE_CMD, RK_TRT_START, payload)

def send_inference_request(payload: dict) -> None:
    """
    Auto-annotation inference 요청 - job_id 기반 동적 바인딩 사용
    """
    job_id = payload.get('job_id')
    if job_id:
        _publish_with_job_binding(EXCHANGE_CMD, RK_INFERENCE_START, payload, job_id)
    else:
        _publish(EXCHANGE_CMD, RK_INFERENCE_START, payload)