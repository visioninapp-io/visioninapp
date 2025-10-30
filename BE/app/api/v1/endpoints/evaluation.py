from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.evaluation import Evaluation
from app.models.model import Model
from app.schemas.evaluation import EvaluationCreate, EvaluationResponse, ModelComparisonResponse

router = APIRouter()


@router.get("/", response_model=List[EvaluationResponse])
async def get_evaluations(
    skip: int = 0,
    limit: int = 100,
    model_id: int = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all evaluations"""
    query = db.query(Evaluation)

    if model_id:
        query = query.filter(Evaluation.model_id == model_id)

    evaluations = query.offset(skip).limit(limit).all()
    return evaluations


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new model evaluation"""
    model = db.query(Model).filter(Model.id == evaluation.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # TODO: Implement actual evaluation logic
    # This would run the model on test dataset and calculate metrics

    # For now, create with provided data
    db_evaluation = Evaluation(
        model_id=evaluation.model_id,
        name=evaluation.name,
        description=evaluation.description,
        test_dataset_name=evaluation.test_dataset_name,
        test_dataset_size=evaluation.test_dataset_size,
        precision=0.0,  # Will be calculated
        recall=0.0,
        f1_score=0.0,
        map_50=0.0,
        class_metrics=[],
        created_by=current_user["uid"]
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


@router.get("/model/{model_id}", response_model=List[EvaluationResponse])
async def get_model_evaluations(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all evaluations for a specific model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    evaluations = db.query(Evaluation).filter(Evaluation.model_id == model_id).all()
    return evaluations


@router.get("/model/{model_id}/latest", response_model=EvaluationResponse)
async def get_latest_evaluation(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get the latest evaluation for a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    evaluation = db.query(Evaluation).filter(
        Evaluation.model_id == model_id
    ).order_by(Evaluation.created_at.desc()).first()

    if not evaluation:
        raise HTTPException(status_code=404, detail="No evaluation found for this model")

    return evaluation


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    model_ids: List[int],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Compare multiple models"""
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 models required for comparison")

    models_data = []
    for model_id in model_ids:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Get latest evaluation
        evaluation = db.query(Evaluation).filter(
            Evaluation.model_id == model_id
        ).order_by(Evaluation.created_at.desc()).first()

        models_data.append({
            "model_id": model.id,
            "model_name": model.name,
            "version": model.version,
            "precision": evaluation.precision if evaluation else None,
            "recall": evaluation.recall if evaluation else None,
            "f1_score": evaluation.f1_score if evaluation else None,
            "map_50": evaluation.map_50 if evaluation else None
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
