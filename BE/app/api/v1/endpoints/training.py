from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from app.core.database import get_db, SessionLocal
from app.core.auth import get_current_user, get_current_user_dev
from app.models.training import TrainingJob, TrainingMetric, TrainingStatus
from app.models.model import Model, ModelStatus, ModelFramework
from app.schemas.training import (
    TrainingJobCreate, TrainingJobUpdate, TrainingJobResponse,
    TrainingMetricCreate, TrainingMetricResponse,
    TrainingControlRequest, HyperparameterTuningRequest
)
from app.services.training_service import training_manager

router = APIRouter()


@router.get("/", response_model=List[TrainingJobResponse])
async def get_training_jobs(
    skip: int = 0,
    limit: int = 100,
    status_filter: str = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all training jobs for current user"""
    query = db.query(TrainingJob).filter(TrainingJob.created_by == current_user["uid"])

    if status_filter:
        query = query.filter(TrainingJob.status == status_filter)

    jobs = query.offset(skip).limit(limit).all()
    return jobs


@router.post("/", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Start a new training job"""
    # Check if name already exists
    existing = db.query(TrainingJob).filter(TrainingJob.name == job.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Training job with this name already exists")

    # Create training job
    total_epochs = job.hyperparameters.get("epochs", 100)

    db_job = TrainingJob(
        name=job.name,
        dataset_id=job.dataset_id,
        architecture=job.architecture,
        hyperparameters=job.hyperparameters,
        total_epochs=total_epochs,
        status=TrainingStatus.PENDING,
        created_by=current_user["uid"]
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    # Create associated model
    model = Model(
        name=f"{job.name}_model",
        architecture=job.architecture,
        framework=ModelFramework.PYTORCH,
        status=ModelStatus.TRAINING,
        training_config=job.hyperparameters,
        hyperparameters=job.hyperparameters,
        created_by=current_user["uid"]
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    # Link model to training job
    db_job.model_id = model.id
    db.commit()
    db.refresh(db_job)

    # Start training process in background
    try:
        training_config = {
            'dataset_id': job.dataset_id,
            'architecture': job.architecture,
            'hyperparameters': job.hyperparameters
        }
        training_manager.start_training(db_job.id, training_config, SessionLocal)
    except Exception as e:
        db_job.status = TrainingStatus.FAILED
        db_job.training_logs = f"Failed to start training: {str(e)}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    return db_job


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get a specific training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.put("/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: int,
    job_update: TrainingJobUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update training job status and metrics"""
    db_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Training job not found")

    update_data = job_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_job, field, value)

    db.commit()
    db.refresh(db_job)
    return db_job


@router.post("/{job_id}/control")
async def control_training(
    job_id: int,
    control: TrainingControlRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Control training job (pause, resume, cancel)"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    try:
        if control.action == "pause":
            training_manager.pause_training(job_id)
            message = "Training paused"
        elif control.action == "resume":
            training_manager.resume_training(job_id)
            message = "Training resumed"
        elif control.action == "cancel":
            training_manager.cancel_training(job_id)
            message = "Training cancelled"
        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        return {"message": message, "status": job.status.value}

    except ValueError as e:
        # Job not in active jobs - update database only
        if control.action == "pause":
            job.status = TrainingStatus.PAUSED
            message = "Training paused"
        elif control.action == "resume":
            job.status = TrainingStatus.RUNNING
            message = "Training resumed"
        elif control.action == "cancel":
            job.status = TrainingStatus.CANCELLED
            message = "Training cancelled"

        db.commit()
        return {"message": message, "status": job.status.value}


@router.get("/{job_id}/metrics", response_model=List[TrainingMetricResponse])
async def get_training_metrics(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get training metrics history"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    metrics = db.query(TrainingMetric).filter(TrainingMetric.training_job_id == job_id).all()
    return metrics


@router.post("/metrics", response_model=TrainingMetricResponse, status_code=status.HTTP_201_CREATED)
async def create_training_metric(
    metric: TrainingMetricCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Add a training metric data point"""
    db_metric = TrainingMetric(**metric.dict())
    db.add(db_metric)

    # Update training job with latest metrics
    job = db.query(TrainingJob).filter(TrainingJob.id == metric.training_job_id).first()
    if job:
        job.current_epoch = metric.epoch
        job.current_loss = metric.train_loss
        job.current_accuracy = metric.train_accuracy
        job.current_learning_rate = metric.learning_rate
        job.progress_percentage = (metric.epoch / job.total_epochs) * 100

        # Update metrics history
        if not job.metrics_history:
            job.metrics_history = {}

        job.metrics_history[str(metric.epoch)] = {
            "loss": metric.train_loss,
            "accuracy": metric.train_accuracy,
            "val_loss": metric.val_loss,
            "val_accuracy": metric.val_accuracy
        }

    db.commit()
    db.refresh(db_metric)
    return db_metric


@router.post("/hyperparameter-tuning")
async def start_hyperparameter_tuning(
    request: HyperparameterTuningRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Start hyperparameter tuning process"""
    # TODO: Implement hyperparameter tuning with Optuna or similar
    # This would create multiple training jobs with different hyperparameters

    return {
        "message": "Hyperparameter tuning started",
        "dataset_id": request.dataset_id,
        "architecture": request.architecture,
        "n_trials": request.n_trials,
        "search_space": request.search_space
    }


@router.get("/{job_id}/progress")
async def get_training_progress(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get real-time training progress"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {
        "job_id": job.id,
        "status": job.status.value,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "progress_percentage": job.progress_percentage,
        "current_loss": job.current_loss,
        "current_accuracy": job.current_accuracy,
        "estimated_completion": job.estimated_completion,
        "metrics_history": job.metrics_history
    }
