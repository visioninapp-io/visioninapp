import os
from mq import publish, publish_with_job_binding
import pika
from pika.exceptions import StreamLostError, ChannelWrongStateError


RABBITMQ_URL = os.getenv("RABBITMQ_URL")


def _safe_publish(ch, ex, routing_key, body, job_id=None):
    """
    job_id 기반으로 동적 바인딩 후 publish.
    끊어졌으면 새 connection/channel 열어서 한 번 더 시도.

    Args:
        ch: RabbitMQ channel
        ex: Exchange name
        routing_key: Routing key
        body: Message body
        job_id: Job ID (optional) - 있으면 동적 바인딩 사용
    """
    try:
        if job_id:
            # job_id가 있으면 동적 바인딩 사용
            publish_with_job_binding(ch, ex, routing_key, body, job_id)
        else:
            # job_id가 없으면 기본 publish 사용
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

            if job_id:
                publish_with_job_binding(new_ch, ex, routing_key, body, job_id)
            else:
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
        _safe_publish(self.ch, self.ex, f"job.{self.job_id}.progress.{stage}", body, job_id=self.job_id)

    def done(self, artifact: dict, metrics: dict | None = None):
        body = {
            "job_id": self.job_id,
            "event": "done",
            "artifact": artifact,
            "metrics": metrics or {},
        }
        _safe_publish(self.ch, self.ex, f"job.{self.job_id}.done", body, job_id=self.job_id)
        print(f"[progress done] send message to {self.job_id} = ..., event: done, artifact: {artifact}, metrics: {metrics or {}}")

    def error(self, stage: str, message: str):
        body = {
            "job_id": self.job_id,
            "event": "error",
            "stage": stage,
            "message": message,
        }
        try:
            _safe_publish(self.ch, self.ex, f"job.{self.job_id}.error", body, job_id=self.job_id)
        except Exception:
            # 에러 전송도 실패하면 더 이상 물고 늘어지지 않고 로그만 남기고 넘겨도 됨
            print(f"[progress error] failed to publish error event for {self.job_id}: {message}")

    def train_log(self, epoch: int, metrics: dict | None = None):
        """
        1 epoch마다 학습 로그 전송.
        routing_key: train.{job_id}.log
        body:
        {
          "job_id": ...,
          "epoch": <int>,
          "metrics": { ... }   # 모든 메트릭 그대로
        }
        """
        body = {
            "job_id": self.job_id,
            "epoch": int(epoch),
            "metrics": metrics or {},
        }
        _safe_publish(self.ch, self.ex, f"train.{self.job_id}.log", body, job_id=self.job_id)

    def train_llm_log(self, epoch: int , total_epochs: int | None = None):
        """
        1 epoch마다 학습 로그 전송.
        routing_key: train.llm.{job_id}.log
        body:
        {
          "job_id": ...,
          "epoch": <int>,
          "total_epochs": <int>,
          "percentage": <int>
        }
        """
        if total_epochs and total_epochs > 0:
            percentage = (int(epoch) / int(total_epochs)) * 70 + 20
        else:
            percentage = 0.0
        body = {
            "job_id": self.job_id,
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "percentage": int(percentage)
        }
        _safe_publish(self.ch, self.ex, f"train.llm.{self.job_id}.log", body, job_id=self.job_id)