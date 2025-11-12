import pika
import json

# ✅ RabbitMQ 연결정보 (하드코딩)
RABBITMQ_URL = "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F"
EXCHANGE = "jobs.events"
ROUTING_KEY = "train.log"

def main():
    print(f"[+] Connecting to {RABBITMQ_URL}")
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    # 익스체인지 선언 (존재해도 OK)
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)

    # 임시 큐 생성
    q = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    qname = q.method.queue

    # train.log 바인딩
    ch.queue_bind(queue=qname, exchange=EXCHANGE, routing_key=ROUTING_KEY)

    print(f"[*] Waiting for messages on '{ROUTING_KEY}' ...\n")

    def callback(ch, method, properties, body):
        try:
            msg = json.loads(body.decode("utf-8"))
            print(f"✅ [train.log] {json.dumps(msg, indent=2, ensure_ascii=False)}\n")
        except Exception as e:
            print(f"⚠️ Invalid message: {e}, raw={body}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=qname, on_message_callback=callback, auto_ack=False)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        print("\n[!] Stopped.")
        conn.close()

if __name__ == "__main__":
    main()
