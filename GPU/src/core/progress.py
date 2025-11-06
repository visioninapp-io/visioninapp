from mq import publish
class Progress:
    def __init__(self, ch, events_exchange: str, job_id: str):
        self.ch = ch
        self.ex = events_exchange
        self.job_id = job_id

    def send(self, stage: str, percent: float, message: str = ""):
        publish(self.ch, self.ex, f"job.{self.job_id}.progress.{stage}",
                {"job_id": self.job_id, "event": "progress", "stage": stage,
                 "percent": float(percent), "message": message})

    def done(self, artifact: dict, metrics: dict | None = None):
        publish(self.ch, self.ex, f"job.{self.job_id}.done",
                {"job_id": self.job_id, "event": "done",
                 "artifact": artifact, "metrics": metrics or {}})

    def error(self, stage: str, message: str):
        publish(self.ch, self.ex, f"job.{self.job_id}.error",
                {"job_id": self.job_id, "event": "error",
                 "stage": stage, "message": message})
