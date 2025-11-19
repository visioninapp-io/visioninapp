# GPU/src/mq.py

import json, threading, logging
import pika
from typing import Callable

logger = logging.getLogger(__name__)


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
    ch.queue_declare(queue=queues["inference"], durable=True)

    ch.queue_bind(queue=queues["train"], exchange=exchanges["cmd"], routing_key="train.start")
    ch.queue_bind(queue=queues["onnx"],  exchange=exchanges["cmd"], routing_key="onnx.start")
    ch.queue_bind(queue=queues["trt"],   exchange=exchanges["cmd"], routing_key="trt.start")
    ch.queue_bind(queue=queues["inference"], exchange=exchanges["cmd"], routing_key="inference.start")


def publish(ch, ex: str, routing_key: str, body: dict):
    """
    기본 publish 함수 (job_id가 없는 경우)
    """
    ch.basic_publish(
        exchange=ex,
        routing_key=routing_key,
        body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )


def publish_with_job_binding(ch, ex: str, routing_key: str, body: dict, job_id: str):
    """
    job_id 기반으로 동적 exchange, queue 선언 및 바인딩 후 publish, 그리고 즉시 해제

    Args:
        ch: RabbitMQ channel
        ex: Exchange name (e.g., 'jobs.events')
        routing_key: Routing key (e.g., 'train.{job_id}.log')
        body: Message body (dict)
        job_id: Job ID for queue naming
    """
    # 1. job_id 기반 queue 이름 생성
    queue_name = f"job.{job_id}.temp.queue"

    try:
        # 2. Exchange 선언 (이미 존재하면 무시됨)
        ch.exchange_declare(exchange=ex, exchange_type="topic", durable=True)
        logger.info(f"[mq] Declared exchange: {ex}")

        # 3. Queue 선언 (임시)
        ch.queue_declare(queue=queue_name, durable=False, auto_delete=True)
        logger.info(f"[mq] Declared temp queue: {queue_name}")

        # 4. Exchange와 Queue를 routing_key로 바인딩
        ch.queue_bind(queue=queue_name, exchange=ex, routing_key=routing_key)
        logger.info(f"[mq] Bound queue '{queue_name}' to exchange '{ex}' with routing_key '{routing_key}'")

        # 5. Publish 메시지
        ch.basic_publish(
            exchange=ex,
            routing_key=routing_key,
            body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        logger.info(f"[mq] Published message to {ex}/{routing_key}")

    finally:
        # 6. 바인딩 해제 및 Queue 삭제
        try:
            ch.queue_unbind(queue=queue_name, exchange=ex, routing_key=routing_key)
            logger.info(f"[mq] Unbound queue '{queue_name}' from exchange '{ex}'")
        except Exception as e:
            logger.warning(f"[mq] Failed to unbind queue {queue_name}: {e}")

        try:
            ch.queue_delete(queue=queue_name)
            logger.info(f"[mq] Deleted temp queue: {queue_name}")
        except Exception as e:
            logger.warning(f"[mq] Failed to delete queue {queue_name}: {e}")


def start_consumer_thread(mq: MQ, queue: str, handler: Callable[[dict], None]):
    """
    - 메시지 받으면 바로 ack (at-most-once).
    - handler는 오래 걸려도 ok (별도 publish 커넥션 사용).
    - 연결 끊기면 while 루프에서 새 conn/ch 열고 자동 재접속.
    """
    def _run():
        while True:
            conn = ch = None
            try:
                # 1) 새 연결
                conn, ch = mq.channel()
                logger.info(f"[mq] consumer connected to {queue}")

                # 2) 콜백 정의
                def _cb(ch_, method, props, body):
                    # (1) 메시지 디코딩
                    try:
                        msg = json.loads(body.decode("utf-8"))
                    except Exception:
                        logger.exception(f"[mq] invalid JSON on {queue}, drop")
                        ch_.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        return

                    # (2) 먼저 ack (재시도 대신 at-most-once 전략)
                    try:
                        ch_.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as e:
                        logger.warning(f"[mq] basic_ack failed on {queue}: {e}")
                        return

                    # (3) 실제 작업 실행 (train/onnx/trt)
                    try:
                        handler(msg)
                    except Exception:
                        logger.exception(f"[mq] handler error on {queue} (message already acked)")

                ch.basic_consume(queue=queue, on_message_callback=_cb)

                # 3) consume 시작 (여기서 block)
                ch.start_consuming()

            except pika.exceptions.StreamLostError as e:
                # 🔥 여기서만 "Reconnecting..." 찍고,
                # while True 때문에 자동으로 새 conn/ch 시도
                logger.warning(f"[mq] consumer connection lost on {queue}: {e}. Reconnecting...")
                continue

            except pika.exceptions.AMQPError as e:
                logger.exception(f"[mq] AMQP error on {queue}: {e}. Reconnecting...")
                continue

            except Exception as e:
                # 예기치 않은 심각한 에러면 멈출지/재시도할지 선택인데,
                # 일단 로그 찍고 재시도하게 놔두고 싶으면 continue
                logger.exception(f"[mq] consumer fatal on {queue}: {e}. Reconnecting...")
                continue

            finally:
                if ch is not None:
                    try:
                        ch.close()
                    except Exception:
                        pass
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
