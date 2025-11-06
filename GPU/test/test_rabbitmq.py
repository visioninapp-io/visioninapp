
# test_rabbitmq_push_threadsafe.py
import os, json, time, uuid, threading, sys
import pika

AMQP_URL   = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
EXCHANGE    = os.getenv("RMQ_TEST_EXCHANGE", "jobs.cmd")
ROUTING_KEY = os.getenv("RMQ_TEST_RK", "health.ping")
QUEUE       = os.getenv("RMQ_TEST_QUEUE", "rmq.health.q")
TIMEOUT     = float(os.getenv("RMQ_TEST_TIMEOUT", "5.0"))

def mk_params(url: str) -> pika.URLParameters:
    p = pika.URLParameters(url)
    # 안정성 옵션 (원하면 조정)
    p.heartbeat = 60
    p.blocked_connection_timeout = 30
    p.socket_timeout = 10
    p.connection_attempts = 5
    p.retry_delay = 2
    return p

def consumer_thread(url, exchange, queue, routing_key, corr_id, done_evt, result):
    conn = pika.BlockingConnection(mk_params(url))
    ch = conn.channel()
    ch.basic_qos(prefetch_count=1)

    # 토폴로지 선언 (멱등)
    ch.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
    ch.queue_declare(queue=queue, durable=True)
    ch.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)

    def on_msg(ch_, method, props, body):
        try:
            data = json.loads(body.decode("utf-8"))
            ok = (
                props.content_type == "application/json" and
                props.correlation_id == corr_id and
                isinstance(data, dict) and
                data.get("correlation_id") == corr_id and
                data.get("test") is True
            )
            if ok:
                ch_.basic_ack(method.delivery_tag)
                result["ok"] = True
            else:
                ch_.basic_nack(method.delivery_tag, requeue=False)
                result["err"] = "content mismatch"
        except Exception as e:
            ch_.basic_nack(method.delivery_tag, requeue=False)
            result["err"] = str(e)
        finally:
            done_evt.set()

    ch.basic_consume(queue=queue, on_message_callback=on_msg)

    try:
        ch.start_consuming()
    except Exception as e:
        # 소비 루프에서의 예외도 결과로 전달
        if not done_evt.is_set():
            result["err"] = f"consumer error: {e!r}"
            done_evt.set()
    finally:
        if conn.is_open:
            try:
                ch.close()
            except Exception:
                pass
            conn.close()

def main():
    corr_id = str(uuid.uuid4())
    got_evt = threading.Event()
    result = {"ok": False, "err": None}

    # 1) 컨슈머는 자기 전용 연결/채널로 별도 스레드에서 실행
    t = threading.Thread(
        target=consumer_thread,
        args=(AMQP_URL, EXCHANGE, QUEUE, ROUTING_KEY, corr_id, got_evt, result),
        daemon=True
    )
    t.start()

    # 2) 퍼블리셔도 자기 전용 연결/채널 사용 (동일 채널 공유 금지!)
    pub_conn = pika.BlockingConnection(mk_params(AMQP_URL))
    pub_ch = pub_conn.channel()
    pub_ch.basic_qos(prefetch_count=1)

    # 토폴로지는 퍼블리셔 쪽에서도 (다중 인스턴스 고려 시 안전)
    pub_ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    pub_ch.queue_declare(queue=QUEUE, durable=True)
    pub_ch.queue_bind(exchange=EXCHANGE, queue=QUEUE, routing_key=ROUTING_KEY)

    # 퍼블리셔 confirm + unroutable 감지
    pub_ch.confirm_delivery()
    unroutable = {"flag": False, "reason": None}
    def on_return(ch_, method, props, body):
        unroutable["flag"] = True
        unroutable["reason"] = f"{method.reply_code} {method.reply_text}"
    pub_ch.add_on_return_callback(on_return)

    payload = {
        "test": True,
        "message": "health check",
        "ts": time.time(),
        "correlation_id": corr_id
    }
    props = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
        correlation_id=corr_id
    )

    # mandatory=True로 라우팅 실패 감지
    pub_ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY,
        body=json.dumps(payload).encode("utf-8"),
        properties=props,
        mandatory=True
    )

    # 발행 confirm 확인
    try:
        pub_ch.wait_for_confirms()
    except Exception as e:
        print(f"[FAIL] publisher confirm error: {e}")
        pub_conn.close()
        # 컨슈머 루프 종료 요청
        # (소비 스레드가 아직 block이면 안전하게 깨우기 위해 stop_consuming 호출)
        try:
            # stop은 컨슈머 쪽 연결/채널에서 해야 하므로 여기선 이벤트만 알리고 종료
            pass
        finally:
            sys.exit(2)

    if unroutable["flag"]:
        print(f"[FAIL] unroutable (no queue bound): {unroutable['reason']}")
        pub_conn.close()
        sys.exit(3)

    # 3) 수신 대기
    if got_evt.wait(TIMEOUT) and result["ok"]:
        print("[PASS] published, routed, and consumed via push (thread-safe)")
        pub_conn.close()
        sys.exit(0)
    else:
        print("[FAIL]", result["err"] or f"no message within {TIMEOUT}s")
        pub_conn.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
