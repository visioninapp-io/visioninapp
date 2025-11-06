import json, logging
from pika import BasicProperties
from app.rabbitmq.connection import get_channel
from app.core.config import settings

log = logging.getLogger(__name__)

def send_train_request(payload: dict) -> None:
    conn, ch = get_channel()
    try:
        body = json.dumps(payload, ensure_ascii=False)
        props = BasicProperties(content_type="application/json", delivery_mode=2)
        ok = ch.basic_publish(
            exchange="",
            routing_key=settings.TRAIN_REQUEST_QUEUE,
            body=body,
            properties=props,
            mandatory=True,
        )
        if not ok:
            raise RuntimeError("Publish not confirmed")
        log.info(f"[RMQ] published job_id={payload.get('job_id')} model={payload.get('hyperparams', {}).get('model')}")
    finally:
        try: ch.close()
        finally: conn.close()
