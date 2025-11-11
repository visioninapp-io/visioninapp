import os
from mq import publish
import pika
from pika.exceptions import StreamLostError, ChannelWrongStateError


RABBITMQ_URL = os.getenv("RABBITMQ_URL")


def _safe_publish(ch, ex, routing_key, body):
    """
    기본 channel publish 시도.
    끊어졌으면 새 connection/channel 열어서 한 번 더 시도.
    두 번째도 실패하면 예외는 올려보내되, 최소한 convert 로직은 끝나도록 설계 가능.
    """
    try:
        publish(ch, ex, routing_key, body)
        return
    except (StreamLostError, ChannelWrongStateError):
        # 기존 채널이 죽은 상태. 새 연결로 재시도.
        if not RABBITMQ_URL:
            # 환경설정 없으면 그냥 실패
            raise

        try:
            params = pika.URLParameters(RABBITMQ_URL)
            conn = pika.BlockingConnection(params)
            new_ch = conn.channel()
            publish(new_ch, ex, routing_key, body)
            new_ch.close()
            conn.close()
        except Exception:
            # 여기서도 실패하면 어차피 MQ 쪽 문제라 더 이상 막지 않고 올림 or 무시 선택 가능
            raise


class Progress:
    def __init__(self, ch, events_exchange: str, job_id: str):
        self.ch = ch
        self.ex = events_exchange
        self.job_id = job_id

    def send(self, stage: str, percent: float, message: str = ""):
        body = {
            "job_id": self.job_id,
            "event": "progress",
            "stage": stage,
            "percent": float(percent),
            "message": message,
        }
        _safe_publish(self.ch, self.ex, f"job.{self.job_id}.progress.{stage}", body)

    def done(self, artifact: dict, metrics: dict | None = None):
        body = {
            "job_id": self.job_id,
            "event": "done",
            "artifact": artifact,
            "metrics": metrics or {},
        }
        _safe_publish(self.ch, self.ex, f"job.{self.job_id}.done", body)
        print(f"[progress done] send message to {self.job_id} = ..., event: done, artifact: {artifact}, metrics: {metrics or {}}")

    def error(self, stage: str, message: str):
        body = {
            "job_id": self.job_id,
            "event": "error",
            "stage": stage,
            "message": message,
        }
        try:
            _safe_publish(self.ch, self.ex, f"job.{self.job_id}.error", body)
        except Exception:
            # 에러 전송도 실패하면 더 이상 물고 늘어지지 않고 로그만 남기고 넘겨도 됨
            print(f"[progress error] failed to publish error event for {self.job_id}: {message}")
