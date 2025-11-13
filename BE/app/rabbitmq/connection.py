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

    # Exchanges
    ch.exchange_declare(exchange="jobs.cmd", exchange_type="topic", durable=True)
    ch.exchange_declare(exchange="jobs.events", exchange_type="topic", durable=True)

    # GPU queues (GPU에서 consume)
    ch.queue_declare(queue="gpu.train.q", durable=True)
    ch.queue_declare(queue="gpu.onnx.q", durable=True)
    ch.queue_declare(queue="gpu.trt.q", durable=True)
    ch.queue_declare(queue="gpu.inference.q", durable=True)

    # Training log queue (Frontend Web STOMP에서 consume - 실시간 metrics)
    ch.queue_declare(queue="gpu.train.log", durable=True)

    # BE queues (BE에서 consume - inference 결과 처리)
    ch.queue_declare(queue="be.inference.done", durable=True)

    # Bindings for cmd exchange (commands to GPU workers)
    ch.queue_bind(exchange="jobs.cmd", queue="gpu.train.q", routing_key="train.start")
    ch.queue_bind(exchange="jobs.cmd", queue="gpu.onnx.q", routing_key="onnx.start")
    ch.queue_bind(exchange="jobs.cmd", queue="gpu.trt.q", routing_key="trt.start")
    ch.queue_bind(exchange="jobs.cmd", queue="gpu.inference.q", routing_key="inference.start")

    # Bindings for events exchange
    ch.queue_bind(exchange="jobs.events", queue="be.inference.done", routing_key="inference.done")
    ch.queue_bind(exchange="jobs.events", queue="gpu.train.log", routing_key="train.log")

    ch.confirm_delivery()
    ch.basic_qos(prefetch_count=10)
    return conn, ch