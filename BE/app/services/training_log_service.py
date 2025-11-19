"""
Training log service for handling train.log events from GPU server
Updates training progress in real-time (each epoch)
"""
import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.training import TrainingJob

log = logging.getLogger(__name__)


def handle_training_log(payload: dict):
    """
    train.log 이벤트 처리 (각 epoch마다 실시간 업데이트)
    
    Expected payload:
    {
        "job_id": "uuid",  # external_job_id (hyperparameters에 저장된 값)
        "epoch": 0,  # 0-based epoch (0 = first epoch completed)
        "metrics": {
            "loss": 38.22,
            "metrics/precision(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            ...
        }
    }
    """
    job_id = payload.get("job_id")  # external_job_id
    epoch = payload.get("epoch", 0)
    metrics = payload.get("metrics", {})

    if not job_id:
        log.warning("[TRAIN-LOG] Missing job_id in payload")
        return

    # DB 세션 생성
    db = SessionLocal()

    try:
        # external_job_id로 TrainingJob 찾기
        all_jobs = db.query(TrainingJob).all()
        job = None
        
        for j in all_jobs:
            if j.hyperparameters and j.hyperparameters.get("external_job_id") == job_id:
                job = j
                break

        if not job:
            # 매칭되는 job이 없으면 경고만 하고 넘어감 (너무 많은 로그 방지)
            # log.debug(f"[TRAIN-LOG] TrainingJob not found for external_job_id={job_id}")
            return

        # Extract metrics
        loss = metrics.get("loss") or metrics.get("tloss", [0])[0] if isinstance(metrics.get("tloss"), list) else metrics.get("tloss", 0)
        
        # mAP50(B) 또는 precision(B)를 accuracy로 사용
        accuracy = (
            metrics.get("metrics/mAP50(B)") or 
            metrics.get("metrics/precision(B)") or 
            metrics.get("metrics/mAP50-95(B)") or 
            0
        )
        
        # Convert accuracy to percentage (if it's in 0-1 range)
        if accuracy < 1:
            accuracy = accuracy * 100

        # Update job metrics
        job.current_epoch = epoch
        job.current_loss = float(loss) if loss else None
        job.current_accuracy = float(accuracy) if accuracy else None
        
        # Calculate progress percentage
        if job.total_epochs and job.total_epochs > 0:
            # epoch is 0-based: epoch 0 = 1st epoch completed
            # progress = (completed_epochs / total_epochs) * 100
            job.progress_percentage = ((epoch + 1) / job.total_epochs) * 100
        
        # Store metrics in history (optional, for historical tracking)
        if not job.metrics_history:
            job.metrics_history = {}
        
        # Store metrics by epoch (convert epoch to string for JSON key)
        job.metrics_history[str(epoch)] = {
            "loss": float(loss) if loss else None,
            "accuracy": float(accuracy) if accuracy else None,
            "raw_metrics": {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in metrics.items() 
                           if isinstance(v, (int, float, str))}
        }

        db.commit()
        
        log.info(
            f"[TRAIN-LOG] Updated job {job.id} ({job.name}): "
            f"epoch={epoch+1}/{job.total_epochs}, "
            f"loss={loss:.4f if loss else 'N/A'}, "
            f"accuracy={accuracy:.2f if accuracy else 'N/A'}%, "
            f"progress={job.progress_percentage:.1f if job.progress_percentage else 0}%"
        )

    except Exception as e:
        log.error(f"[TRAIN-LOG] Error processing job_id={job_id}, epoch={epoch}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

