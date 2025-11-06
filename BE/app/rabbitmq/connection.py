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
    ch.queue_declare(queue=settings.TRAIN_REQUEST_QUEUE, durable=True)
    ch.confirm_delivery()
    ch.basic_qos(prefetch_count=10)
    return conn, ch
