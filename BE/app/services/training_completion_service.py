"""
Training completion service for handling train.done events from GPU server
"""
import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.training import TrainingJob, TrainingStatus
from app.utils.timezone import get_kst_now_naive

log = logging.getLogger(__name__)


def handle_training_done(payload: dict):
    """
    train.done 이벤트 처리

    Expected payload:
    {
        "job_id": "uuid",  # external_job_id (hyperparameters에 저장된 값)
        "status": "completed" | "failed",
        "error_message": null | "error details",
        "metrics": {
            "final_loss": 0.123,
            "final_accuracy": 0.95,
            ...
        }
    }
    """
    job_id = payload.get("job_id")  # external_job_id
    status = payload.get("status", "completed")
    error_message = payload.get("error_message")
    metrics = payload.get("metrics", {})

    log.info(f"[TRAIN-DONE] Processing job_id={job_id}, status={status}")

    if not job_id:
        log.error("[TRAIN-DONE] Missing job_id in payload")
        return

    # DB 세션 생성
    db = SessionLocal()

    try:
        # external_job_id로 TrainingJob 찾기
        # hyperparameters JSON 필드에서 external_job_id 검색
        all_jobs = db.query(TrainingJob).all()
        job = None
        
        for j in all_jobs:
            if j.hyperparameters and j.hyperparameters.get("external_job_id") == job_id:
                job = j
                break

        if not job:
            log.warning(f"[TRAIN-DONE] TrainingJob not found for external_job_id={job_id}")
            return

        log.info(f"[TRAIN-DONE] Found TrainingJob id={job.id}, name={job.name}")

        # 상태 업데이트
        if status == "completed":
            job.status = TrainingStatus.COMPLETED
            job.completed_at = get_kst_now_naive()
            
            # 메트릭 업데이트 (있는 경우)
            if metrics:
                if "final_loss" in metrics:
                    job.current_loss = metrics.get("final_loss")
                if "final_accuracy" in metrics:
                    job.current_accuracy = metrics.get("final_accuracy")
            
            job.training_log = f"Training completed successfully. Metrics: {metrics}" if metrics else "Training completed successfully"
            log.info(f"[TRAIN-DONE] Updated job {job.id} to COMPLETED")
            
        elif status == "failed":
            job.status = TrainingStatus.FAILED
            job.error_message = error_message or "Training failed"
            job.training_log = f"Training failed: {error_message}" if error_message else "Training failed"
            log.warning(f"[TRAIN-DONE] Updated job {job.id} to FAILED: {error_message}")
        else:
            log.warning(f"[TRAIN-DONE] Unknown status: {status}")

        db.commit()
        log.info(f"[TRAIN-DONE] Successfully updated TrainingJob id={job.id}")

    except Exception as e:
        log.error(f"[TRAIN-DONE] Error processing training completion: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

