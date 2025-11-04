import json, threading
import pika
from typing import Callable, Optional

class MQ:
    def __init__(self, amqp_url: str):
        self.params = pika.URLParameters(amqp_url)

    def channel(self):
        conn = pika.BlockingConnection(self.params)
        ch = conn.channel()
        ch.basic_qos(prefetch_count=1)
        return conn, ch

def declare_topology(ch, exchanges: dict, queues: dict):
    ch.exchange_declare(exchange=exchanges["cmd"], exchange_type="topic", durable=True)
    ch.exchange_declare(exchange=exchanges["events"], exchange_type="topic", durable=True)
    ch.queue_declare(queue=queues["train"], durable=True)
    ch.queue_declare(queue=queues["onnx"], durable=True)
    ch.queue_declare(queue=queues["trt"], durable=True)
    ch.queue_bind(exchange=exchanges["cmd"], queue=queues["train"], routing_key="train.start")
    ch.queue_bind(exchange=exchanges["cmd"], queue=queues["onnx"],  routing_key="onnx.start")
    ch.queue_bind(exchange=exchanges["cmd"], queue=queues["trt"],   routing_key="trt.start")

def publish(ch, exchange: str, routing_key: str, body: dict):
    ch.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=json.dumps(body).encode("utf-8"),
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2)
    )

def start_consumer_thread(mq: MQ, queue: str, handler: Callable[[dict], None]):
    def _run():
        conn, ch = mq.channel()
        try:
            def _cb(ch_, method, props, body):
                try:
                    msg = json.loads(body.decode("utf-8"))
                    handler(msg)
                    ch_.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    ch_.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            ch.basic_consume(queue=queue, on_message_callback=_cb)
            ch.start_consuming()
        finally:
            conn.close()
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th
