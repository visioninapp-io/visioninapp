import pika, ssl
from app.core.config import settings

def _params():
    creds = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD)
    ssl_opts = None
    if settings.RABBITMQ_SSL:
        ctx = ssl.create_default_context()
        ssl_opts = pika.SSLOptions(ctx, settings.RABBITMQ_HOST)

    return pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host=settings.RABBITMQ_VHOST,
        credentials=creds,
        ssl_options=ssl_opts,
        heartbeat=30,
        blocked_connection_timeout=30,
        connection_attempts=5, retry_delay=2, socket_timeout=10,
    )

def get_channel():
    conn = pika.BlockingConnection(_params())
    ch = conn.channel()
    ch.exchange_declare(exchange="jobs.cmd", exchange_type="topic", durable=True)
    ch.exchange_declare(exchange="jobs.events", exchange_type="topic", durable=True)
    
    # Queue 선언
    ch.queue_declare(queue=settings.TRAIN_REQUEST_QUEUE, durable=True)
    ch.queue_declare(queue="gpu.onnx.q", durable=True)
    ch.queue_declare(queue="gpu.trt.q", durable=True)
    
    # Queue 바인딩
    ch.queue_bind(queue=settings.TRAIN_REQUEST_QUEUE, exchange="jobs.cmd", routing_key="train.start")
    ch.queue_bind(queue="gpu.onnx.q", exchange="jobs.cmd", routing_key="onnx.start")
    ch.queue_bind(queue="gpu.trt.q", exchange="jobs.cmd", routing_key="trt.start")
    
    ch.confirm_delivery()
    ch.basic_qos(prefetch_count=10)
    return conn, ch