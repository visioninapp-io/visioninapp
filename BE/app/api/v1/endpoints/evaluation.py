from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
import boto3
import csv
import io
import re
from botocore.exceptions import ClientError
from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.config import settings
from app.models.evaluation import Evaluation
from app.schemas.evaluation import EvaluationCreate, EvaluationResponse, ModelComparisonResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _slugify_model_name(s: str) -> str:
    """Convert model name to S3-friendly key"""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]", "", s)
    return s or "model"


@router.get("/", response_model=List[EvaluationResponse])
async def get_evaluations(
    skip: int = 0,
    limit: int = 10000,
    model_version_id: int = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all evaluations"""
    query = db.query(Evaluation)

    if model_version_id:
        query = query.filter(Evaluation.model_version_id == model_version_id)

    evaluations = query.offset(skip).limit(limit).all()
    return evaluations


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new model evaluation"""
    from app.models.model_version import ModelVersion
    
    # ModelVersion 존재 확인
    model_version = db.query(ModelVersion).filter(ModelVersion.id == evaluation.model_version_id).first()
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Evaluation 생성
    db_evaluation = Evaluation(
        model_version_id=evaluation.model_version_id,
        f1_score=evaluation.f1_score,
        precision=evaluation.precision,
        recall=evaluation.recall,
        mAP_50=evaluation.mAP_50,
        mAP_50_95=evaluation.mAP_50_95
    )
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)

    return db_evaluation


@router.get("/completed-trainings", response_model=List[Dict[str, Any]])
async def get_completed_trainings(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get completed training jobs with their models for evaluation"""
    from app.models.training import TrainingJob, TrainingStatus
    from app.models.model import Model
    
    # 완료된 학습만 가져오기
    completed_jobs = db.query(TrainingJob).filter(
        TrainingJob.status == TrainingStatus.COMPLETED.value,
        TrainingJob.created_by == current_user["uid"]
    ).order_by(TrainingJob.created_at.desc()).all()
    
    result = []
    for job in completed_jobs:
        model = db.query(Model).filter(Model.id == job.model_id).first() if job.model_id else None
        
        # model_key 추출 (S3 경로에 사용)
        # training.py와 동일한 방식으로 생성
        if model:
            model_name = model.name
        else:
            # model이 없으면 job.name을 사용
            model_name = f"{job.name}_model"
        
        model_key = _slugify_model_name(model_name)
        
        result.append({
            "id": job.id,
            "name": job.name,
            "model_id": job.model_id,
            "model_name": model_name,
            "model_key": model_key,
            "architecture": job.architecture,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "has_results": True  # results.csv 존재 여부는 실제로 확인해야 하지만 일단 True
        })
    
    return result


@router.get("/results/{model_key}")
async def get_evaluation_results(
    model_key: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get training results CSV from S3 for evaluation
    Returns parsed CSV data as JSON
    Path: s3://bucket/models/{model_key}/results.csv
    """
    try:
        logger.info(f"[Evaluation] Fetching results.csv for model_key: {model_key}")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # S3 key for results.csv
        s3_key = f"models/{model_key}/results.csv"

        logger.info(f"[Evaluation] Downloading from s3://{settings.AWS_BUCKET_NAME}/{s3_key}")

        # Download file from S3
        response = s3_client.get_object(
            Bucket=settings.AWS_BUCKET_NAME,
            Key=s3_key
        )

        # Read and parse CSV content
        csv_content = response['Body'].read().decode('utf-8')
        
        # Parse CSV to JSON
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)
        
        # Convert numeric strings to numbers
        for row in rows:
            for key, value in row.items():
                try:
                    # Try to convert to float
                    row[key] = float(value)
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    pass

        logger.info(f"[Evaluation] Successfully parsed {len(rows)} rows from results.csv")

        return {
            "model_key": model_key,
            "rows": rows,
            "columns": list(rows[0].keys()) if rows else []
        }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            logger.warning(f"[Evaluation] File not found: s3://{settings.AWS_BUCKET_NAME}/{s3_key}")
            raise HTTPException(
                status_code=404,
                detail=f"Results file not found for model '{model_key}'. The training may not have completed yet."
            )
        else:
            logger.error(f"[Evaluation] S3 error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")
    except Exception as e:
        logger.error(f"[Evaluation] Error fetching results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific evaluation"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.get("/model-version/{model_version_id}", response_model=List[EvaluationResponse])
async def get_model_version_evaluations(
    model_version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all evaluations for a specific model version"""
    from app.models.model_version import ModelVersion
    
    model_version = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")

    evaluations = db.query(Evaluation).filter(Evaluation.model_version_id == model_version_id).all()
    return evaluations


@router.get("/model-version/{model_version_id}/latest", response_model=EvaluationResponse)
async def get_latest_evaluation(
    model_version_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get the latest evaluation for a model version"""
    from app.models.model_version import ModelVersion
    
    model_version = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")

    evaluation = db.query(Evaluation).filter(
        Evaluation.model_version_id == model_version_id
    ).order_by(Evaluation.id.desc()).first()

    if not evaluation:
        raise HTTPException(status_code=404, detail="No evaluation found for this model version")

    return evaluation


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    model_version_ids: List[int],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Compare multiple model versions"""
    if len(model_version_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 model versions required for comparison")

    from app.models.model_version import ModelVersion
    
    models_data = []
    for model_version_id in model_version_ids:
        model_version = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not model_version:
            raise HTTPException(status_code=404, detail=f"Model version {model_version_id} not found")

        # Get latest evaluation
        evaluation = db.query(Evaluation).filter(
            Evaluation.model_version_id == model_version_id
        ).order_by(Evaluation.id.desc()).first()

        models_data.append({
            "model_version_id": model_version.id,
            "model_name": model_version.model.name if model_version.model else "Unknown",
            "version_tag": model_version.version_tag,
            "precision": evaluation.precision if evaluation else None,
            "recall": evaluation.recall if evaluation else None,
            "f1_score": evaluation.f1_score if evaluation else None,
            "map_50": evaluation.mAP_50 if evaluation else None
        })

    return {
        "models": models_data,
        "metrics": ["precision", "recall", "f1_score", "map_50"],
        "comparison_chart_data": {
            "labels": [m["model_name"] for m in models_data],
            "datasets": [
                {
                    "label": "Precision",
                    "data": [m["precision"] for m in models_data]
                },
                {
                    "label": "Recall",
                    "data": [m["recall"] for m in models_data]
                },
                {
                    "label": "F1-Score",
                    "data": [m["f1_score"] for m in models_data]
                }
            ]
        }
    }


@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete an evaluation"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    db.delete(evaluation)
    db.commit()
    return None
