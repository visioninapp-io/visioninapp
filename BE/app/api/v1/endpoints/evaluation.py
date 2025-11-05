from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.evaluation import Evaluation
from app.schemas.evaluation import EvaluationCreate, EvaluationResponse, ModelComparisonResponse

router = APIRouter()


@router.get("/", response_model=List[EvaluationResponse])
async def get_evaluations(
    skip: int = 0,
    limit: int = 10000,
    model_version_id: int = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
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


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
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
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete an evaluation"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    db.delete(evaluation)
    db.commit()
    return None
