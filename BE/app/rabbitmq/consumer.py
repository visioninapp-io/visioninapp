import json
import logging
from typing import Callable
from pika import BlockingConnection
from app.rabbitmq.connection import get_channel

log = logging.getLogger(__name__)

EXCHANGE_EVENTS = "jobs.events"
RK_INFERENCE_DONE = "inference.done"
RK_TRAIN_DONE = "job.*.done"  # 와일드카드: job.{job_id}.done 매칭
RK_TRAIN_LOG = "train.log.*"  # 와일드카드: train.log.{job_id} 매칭
RK_TRAIN_LLM_LOG = "train.llm.log.*"  # 와일드카드: train.llm.log.{job_id} 매칭
RK_TRAIN_LLM_HPO = "train.llm.hpo.*"  # 와일드카드: train.llm.hpo.{job_id} 매칭
RK_ONNX_DONE = "onnx.done"
RK_TRT_DONE = "trt.done"


def start_inference_consumer(handler: Callable[[dict], None]):
    """
    inference.done 이벤트를 소비하는 Consumer 시작

    Args:
        handler: 메시지 처리 함수 (payload dict를 받음)
    """
    conn, ch = get_channel()

    # Queue 선언
    queue_name = "be.inference.done"
    ch.queue_declare(queue=queue_name, durable=True)
    ch.queue_bind(exchange="jobs.events", queue="be.inference.done", routing_key="inference.done")
    # Binding: jobs.events exchange -> inference_done queue (routing_key: inference.done)
    ch.queue_bind(
        exchange=EXCHANGE_EVENTS,
        queue=queue_name,
        routing_key=RK_INFERENCE_DONE
    )

    log.info(f"[RMQ Consumer] Waiting for {RK_INFERENCE_DONE} messages on queue '{queue_name}'...")

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            log.info(f"[RMQ Consumer] Received {RK_INFERENCE_DONE}: job_id={payload.get('job_id')}")

            # Handler 실행
            handler(payload)

            # Ack
            ch.basic_ack(delivery_tag=method.delivery_tag)
            log.info(f"[RMQ Consumer] Processed {RK_INFERENCE_DONE}: job_id={payload.get('job_id')}")

        except Exception as e:
            log.error(f"[RMQ Consumer] Error processing {RK_INFERENCE_DONE}: {e}", exc_info=True)
            # Nack (requeue=False로 DLQ로 이동)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    ch.basic_consume(queue=queue_name, on_message_callback=callback)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        log.info("[RMQ Consumer] Stopping consumer...")
        ch.stop_consuming()
    except Exception as e:
        log.error(f"[RMQ] Consumer error: {e}")
    finally:
        # Close connection safely (only if not already closed)
        try:
            if conn and conn.is_open:
                conn.close()
                log.info("[RMQ Consumer] Connection closed")
        except Exception as e:
            log.warning(f"[RMQ Consumer] Error closing connection: {e}")


def start_training_consumer(handler: Callable[[dict], None]):
    """
    job.*.done 이벤트를 소비하는 Consumer 시작
    와일드카드 사용: job.{job_id}.done 매칭

    Args:
        handler: 메시지 처리 함수 (payload dict를 받음)
    """
    conn, ch = get_channel()

    # Queue 선언
    queue_name = "be.train.done"
    ch.queue_declare(queue=queue_name, durable=True)

    # Binding: jobs.events exchange -> train_done queue (routing_key: job.*.done)
    ch.queue_bind(
        exchange=EXCHANGE_EVENTS,
        queue=queue_name,
        routing_key=RK_TRAIN_DONE  # job.*.done (와일드카드)
    )

    log.info(f"[RMQ Consumer] Waiting for {RK_TRAIN_DONE} messages on queue '{queue_name}'...")

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            log.info(f"[RMQ Consumer] Received {RK_TRAIN_DONE}: job_id={payload.get('job_id')}")

            # Handler 실행
            handler(payload)

            # Ack
            ch.basic_ack(delivery_tag=method.delivery_tag)
            log.info(f"[RMQ Consumer] Processed {RK_TRAIN_DONE}: job_id={payload.get('job_id')}")

        except Exception as e:
            log.error(f"[RMQ Consumer] Error processing {RK_TRAIN_DONE}: {e}", exc_info=True)
            # Nack (requeue=False로 DLQ로 이동)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    ch.basic_consume(queue=queue_name, on_message_callback=callback)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        log.info("[RMQ Consumer] Stopping training consumer...")
        ch.stop_consuming()
    except Exception as e:
        log.error(f"[RMQ] Training consumer error: {e}")
    finally:
        # Close connection safely (only if not already closed)
        try:
            if conn and conn.is_open:
                conn.close()
                log.info("[RMQ Consumer] Training consumer connection closed")
        except Exception as e:
            log.warning(f"[RMQ Consumer] Error closing training consumer connection: {e}")


def start_llm_hpo_consumer(handler: Callable[[dict], None]):
    """
    train.llm.hpo 이벤트를 소비하는 Consumer 시작
    LLM이 결정한 hyperparameters로 DB 업데이트
    와일드카드 사용: job_id별로 구분

    Args:
        handler: 메시지 처리 함수 (payload dict를 받음)
    """
    conn, ch = get_channel()

    # Queue 선언
    queue_name = "be.train.llm.hpo"
    ch.queue_declare(queue=queue_name, durable=True)

    # Binding: jobs.events exchange -> llm_hpo queue (routing_key: train.llm.hpo.*)
    ch.queue_bind(
        exchange=EXCHANGE_EVENTS,
        queue=queue_name,
        routing_key=RK_TRAIN_LLM_HPO  # train.llm.hpo.* (와일드카드)
    )

    log.info(f"[RMQ Consumer] Waiting for {RK_TRAIN_LLM_HPO} messages on queue '{queue_name}'...")

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            log.info(f"[RMQ Consumer] Received {RK_TRAIN_LLM_HPO}: job_id={payload.get('job_id')}")

            # Handler 실행
            handler(payload)

            # Ack
            ch.basic_ack(delivery_tag=method.delivery_tag)
            log.info(f"[RMQ Consumer] Processed {RK_TRAIN_LLM_HPO}: job_id={payload.get('job_id')}")

        except Exception as e:
            log.error(f"[RMQ Consumer] Error processing {RK_TRAIN_LLM_HPO}: {e}", exc_info=True)
            # Nack (requeue=False로 DLQ로 이동)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    ch.basic_consume(queue=queue_name, on_message_callback=callback)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        log.info("[RMQ Consumer] Stopping LLM HPO consumer...")
        ch.stop_consuming()
    except Exception as e:
        log.error(f"[RMQ] LLM HPO consumer error: {e}")
    finally:
        # Close connection safely (only if not already closed)
        try:
            if conn and conn.is_open:
                conn.close()
                log.info("[RMQ Consumer] LLM HPO consumer connection closed")
        except Exception as e:
            log.warning(f"[RMQ Consumer] Error closing LLM HPO consumer connection: {e}")


def start_train_log_consumer(handler: Callable[[dict], None]):
    """
    train.log / train.llm.log 이벤트를 소비하는 Consumer 시작
    각 epoch마다 실시간으로 training progress 업데이트
    와일드카드 사용: job_id별로 구분
    
    Args:
        handler: 메시지 처리 함수 (payload dict를 받음)
    """
    conn, ch = get_channel()

    # Queue 선언
    queue_name = "be.train.log"
    ch.queue_declare(queue=queue_name, durable=True)

    # Binding 1: 일반 트레이닝용 train.log.*
    ch.queue_bind(
        exchange=EXCHANGE_EVENTS,
        queue=queue_name,
        routing_key=RK_TRAIN_LOG  # train.log.* (와일드카드)
    )
    
    # Binding 2: LLM 트레이닝용 train.llm.log.*
    ch.queue_bind(
        exchange=EXCHANGE_EVENTS,
        queue=queue_name,
        routing_key=RK_TRAIN_LLM_LOG  # train.llm.log.* (와일드카드)
    )

    log.info(f"[RMQ Consumer] Waiting for {RK_TRAIN_LOG} messages on queue '{queue_name}'...")

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            # Debug logging (can be disabled for production to reduce log volume)
            # log.debug(f"[RMQ Consumer] Received {RK_TRAIN_LOG}: job_id={payload.get('job_id')}, epoch={payload.get('epoch')}")

            # Handler 실행
            handler(payload)

            # Ack
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            log.error(f"[RMQ Consumer] Error processing {RK_TRAIN_LOG}: {e}", exc_info=True)
            # Ack anyway to prevent message requeue (train.log는 손실되어도 괜찮음)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=queue_name, on_message_callback=callback)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        log.info("[RMQ Consumer] Stopping train log consumer...")
        ch.stop_consuming()
    except Exception as e:
        log.error(f"[RMQ] Train log consumer error: {e}")
    finally:
        # Close connection safely (only if not already closed)
        try:
            if conn and conn.is_open:
                conn.close()
                log.info("[RMQ Consumer] Train log consumer connection closed")
        except Exception as e:
            log.warning(f"[RMQ Consumer] Error closing train log consumer connection: {e}")