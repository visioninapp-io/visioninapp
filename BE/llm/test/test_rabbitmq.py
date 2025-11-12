# consumer_any.py
import pika, json

RABBITMQ_URL = "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F"
EXCHANGE = "jobs.events"
BINDING_KEY = "#"   # ← 모든 라우팅키 수신 (topic)

params = pika.URLParameters(RABBITMQ_URL)
conn = pika.BlockingConnection(params)
ch = conn.channel()

ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
qname = q.method.queue
ch.queue_bind(queue=qname, exchange=EXCHANGE, routing_key=BINDING_KEY)

print(f"[*] Waiting on exchange='{EXCHANGE}', key='{BINDING_KEY}' ...")
def cb(chx, method, props, body):
    try:
        print(f"\n[key={method.routing_key}] {json.loads(body.decode())}")
    except Exception:
        print(f"\n[key={method.routing_key}] raw={body!r}")
    chx.basic_ack(method.delivery_tag)

ch.basic_consume(queue=qname, on_message_callback=cb, auto_ack=False)
try:
    ch.start_consuming()
except KeyboardInterrupt:
    conn.close()
