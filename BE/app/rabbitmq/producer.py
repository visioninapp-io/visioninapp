import json, logging
from pika import BasicProperties
from app.rabbitmq.connection import get_channel

log = logging.getLogger(__name__)

EXCHANGE_CMD = "jobs.cmd"
EXCHANGE_EVENTS = "jobs.events" # Added for consistency if needed later for direct publishing from BE
RK_TRAIN_START = "train.start"
RK_ONNX_START  = "onnx.start"
RK_TRT_START   = "trt.start"
RK_INFERENCE_START = 'inference.start'

def _publish(exchange: str, routing_key: str, payload: dict) -> None:
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
        log.info(f"[RMQ] published rk={routing_key} job_id={payload.get('job_id')}, user_id={payload.get('user_id')}, model_id={payload.get('model_id')}")
    finally:
        try: ch.close()
        finally: conn.close()

def send_train_request(payload: dict, user_id: str, model_id: int) -> None:
    # Train 요청 - user_id와 model_id 추가
    payload['user_id'] = user_id
    payload['model_id'] = model_id
    _publish(EXCHANGE_CMD, RK_TRAIN_START, payload)

def send_onnx_request(payload: dict) -> None:
    # ONNX 변환 요청
    _publish(EXCHANGE_CMD, RK_ONNX_START, payload)

def send_trt_request(payload: dict) -> None:
    # TensorRT 변환 요청
    _publish(EXCHANGE_CMD, RK_TRT_START, payload)

def send_inference_request(payload: dict) -> None:
    # Auto-annotation inference 요청
    _publish(EXCHANGE_CMD, RK_INFERENCE_START,payload)