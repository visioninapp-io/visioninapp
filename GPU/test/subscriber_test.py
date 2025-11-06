# subscriber_test.py
import os, json, pika

AMQP_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
EXCHANGE    = os.getenv("RMQ_EXCHANGE", "jobs.cmd")     # topic exchange
ROUTING_KEY = os.getenv("RMQ_ROUTING_KEY", "health.test")
QUEUE       = os.getenv("RMQ_QUEUE", "rmq.health.q")    # 테스트용 고정 큐

def main():
    params = pika.URLParameters(AMQP_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.basic_qos(prefetch_count=1)

    # 토폴로지(멱등)
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    ch.queue_declare(queue=QUEUE, durable=True)
    ch.queue_bind(exchange=EXCHANGE, queue=QUEUE, routing_key=ROUTING_KEY)

    print(f"[SUB] waiting on queue={QUEUE} (exchange={EXCHANGE}, rk={ROUTING_KEY})")

    def on_msg(ch_, method, props, body):
        try:
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                data = body.decode("utf-8", errors="replace")
            print(f"[SUB] received: props.correlation_id={getattr(props, 'correlation_id', None)} body={data}")
            ch_.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print("[SUB] error while handling message:", e)
            ch_.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    ch.basic_consume(queue=QUEUE, on_message_callback=on_msg)
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        print("\n[SUB] stopping...")
    finally:
        if conn.is_open:
            ch.close()
            conn.close()

if __name__ == "__main__":
    main()
