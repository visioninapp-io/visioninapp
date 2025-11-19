"""
Training HPO service for handling train.llm.hpo events from LLM pipeline
"""
import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.training import TrainingJob

log = logging.getLogger(__name__)


def handle_llm_hpo(payload: dict):
    """
    train.llm.hpo 이벤트 처리
    LLM이 결정한 hyperparameters로 DB 업데이트

    Expected payload:
    {
        "job_id": "uuid",  # external_job_id (hyperparameters에 저장된 값)
        "hyperparams": {
            "epochs": 100,
            "batch": 16,
            "lr0": 0.001,
            ...
        }
    }
    """
    job_id = payload.get("job_id")  # external_job_id
    hyperparams = payload.get("hyperparams", {})

    log.info(f"[LLM-HPO] Processing job_id={job_id}, hyperparams={hyperparams}")

    if not job_id:
        log.warning("[LLM-HPO] Missing job_id in payload")
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
            log.warning(f"[LLM-HPO] TrainingJob not found for external_job_id={job_id}")
            return

        log.info(f"[LLM-HPO] Found TrainingJob id={job.id}, name={job.name}")

        # hyperparameters 업데이트
        if not job.hyperparameters:
            job.hyperparameters = {}
        
        # 기존 hyperparameters에 LLM이 결정한 값 병합
        job.hyperparameters.update(hyperparams)
        
        # total_epochs 업데이트 (epochs 값이 있으면)
        if "epochs" in hyperparams:
            new_total_epochs = int(hyperparams["epochs"])
            if job.total_epochs != new_total_epochs:
                log.info(f"[LLM-HPO] Updating total_epochs: {job.total_epochs} -> {new_total_epochs}")
                job.total_epochs = new_total_epochs
        
        # architecture 업데이트 (model 정보가 있으면)
        if "model" in hyperparams:
            model_name = hyperparams["model"]
            if job.architecture == "AI-Auto":
                log.info(f"[LLM-HPO] Updating architecture: AI-Auto -> {model_name}")
                job.architecture = model_name

        db.commit()
        log.info(f"[LLM-HPO] Successfully updated TrainingJob id={job.id} with hyperparameters")

    except Exception as e:
        log.error(f"[LLM-HPO] Error processing job_id={job_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

