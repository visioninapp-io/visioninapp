"""
Background service for periodically syncing training job status from S3
Runs automatically to catch any missed train.done events
"""
import logging
import time
import boto3
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.config import settings
from app.models.training import TrainingJob, TrainingStatus
from app.models.model import Model
from app.utils.timezone import get_kst_now_naive
from datetime import datetime

log = logging.getLogger(__name__)


def _slugify_model_name(s: str) -> str:
    """Convert model name to S3-friendly key"""
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]", "", s)
    return s or "model"


def sync_completed_trainings():
    """
    Check S3 for results.csv files and automatically mark training jobs as completed
    This is a background task that runs periodically
    """
    db = SessionLocal()
    
    try:
        # Initialize S3 client
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
        except Exception as e:
            log.warning(f"[Training Sync] Failed to initialize S3 client: {e}")
            return
        
        # Get all running/pending training jobs
        running_jobs = db.query(TrainingJob).filter(
            TrainingJob.status.in_([TrainingStatus.RUNNING.value, TrainingStatus.PENDING.value])
        ).all()
        
        if not running_jobs:
            return
        
        log.info(f"[Training Sync] Checking {len(running_jobs)} running/pending jobs for completion")
        
        updated_count = 0
        
        for job in running_jobs:
            try:
                # Get model_key from job
                model = db.query(Model).filter(Model.id == job.model_id).first() if job.model_id else None
                
                if model:
                    model_name = model.name
                else:
                    model_name = f"{job.name}_model"
                
                model_key = _slugify_model_name(model_name)
                s3_key = f"models/{model_key}/results.csv"
                
                # Check if results.csv exists in S3
                try:
                    s3_client.head_object(
                        Bucket=settings.AWS_BUCKET_NAME,
                        Key=s3_key
                    )
                    
                    # results.csv exists, mark as completed
                    if job.status != TrainingStatus.COMPLETED.value:
                        job.status = TrainingStatus.COMPLETED
                        if not job.completed_at:
                            job.completed_at = datetime.utcnow()
                        job.training_log = (job.training_log or "") + f"\nAuto-synced completion (results.csv found in S3 at {datetime.utcnow().isoformat()})"
                        db.commit()
                        
                        updated_count += 1
                        log.info(f"[Training Sync] Auto-marked job {job.id} ({job.name}) as completed - results.csv found")
                    
                except ClientError as e:
                    if e.response.get('Error', {}).get('Code') == '404':
                        # File doesn't exist, skip
                        pass
                    else:
                        log.warning(f"[Training Sync] Error checking S3 for job {job.id}: {e}")
                        
            except Exception as e:
                log.error(f"[Training Sync] Error processing job {job.id}: {e}", exc_info=True)
                continue
        
        if updated_count > 0:
            log.info(f"[Training Sync] Synced {updated_count} training job(s) to completed status")
        
    except Exception as e:
        log.error(f"[Training Sync] Error in sync task: {e}", exc_info=True)
    finally:
        db.close()


def run_periodic_sync(interval_minutes: int = 5):
    """
    Run periodic sync in a background thread
    Checks S3 every interval_minutes for completed trainings
    """
    log.info(f"[Training Sync] Starting periodic sync service (interval: {interval_minutes} minutes)")
    
    while True:
        try:
            time.sleep(interval_minutes * 60)  # Convert minutes to seconds
            sync_completed_trainings()
        except KeyboardInterrupt:
            log.info("[Training Sync] Periodic sync service stopped")
            break
        except Exception as e:
            log.error(f"[Training Sync] Error in periodic sync: {e}", exc_info=True)
            # Continue even if there's an error
            time.sleep(60)  # Wait 1 minute before retrying

