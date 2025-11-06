from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.deployment import Deployment
from app.schemas.deployment import (
    DeploymentCreate, DeploymentUpdate, DeploymentResponse, DeploymentStats,
    InferenceRequest, InferenceResponse, PredictionResult
)

router = APIRouter()


@router.get("/stats", response_model=DeploymentStats)
async def get_deployment_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get overall deployment statistics"""
    # 호환성 필드가 제거되어 기본값 반환
    deployments = db.query(Deployment).all()

    active_deployments = len(deployments)
    
    return {
        "active_deployments": active_deployments,
        "total_requests": 0,  # 호환성 필드 제거됨
        "avg_response_time": 0.0,  # 호환성 필드 제거됨
        "uptime_percentage": 0.0  # 호환성 필드 제거됨
    }


@router.get("/", response_model=List[DeploymentResponse])
async def get_deployments(
    skip: int = 0,
    limit: int = 10000,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all deployments"""
    query = db.query(Deployment)
    deployments = query.offset(skip).limit(limit).all()
    return deployments


@router.post("/", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    deployment: DeploymentCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new deployment"""
    from app.models.model_version import ModelVersion
    
    # Check if model version exists
    model_version = db.query(ModelVersion).filter(ModelVersion.id == deployment.model_version_id).first()
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Check if name already exists
    existing = db.query(Deployment).filter(Deployment.name == deployment.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Deployment with this name already exists")

    # Deployment 생성
    db_deployment = Deployment(
        model_version_id=deployment.model_version_id,
        name=deployment.name,
        target=deployment.target,
        endpoint_url=deployment.endpoint_url
    )
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)

    # TODO: Implement actual deployment process
    # This would deploy to edge device, cloud, etc.

    return db_deployment


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get a specific deployment"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


@router.put("/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(
    deployment_id: int,
    deployment_update: DeploymentUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update a deployment"""
    db_deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not db_deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    update_data = deployment_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        if hasattr(db_deployment, field):
            setattr(db_deployment, field, value)

    db.commit()
    db.refresh(db_deployment)
    return db_deployment


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete a deployment"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # TODO: Stop actual deployment before deleting

    db.delete(deployment)
    db.commit()
    return None


@router.post("/{deployment_id}/inference", response_model=InferenceResponse)
async def run_inference(
    deployment_id: int,
    request: InferenceRequest,
    db: Session = Depends(get_db)
):
    """Run inference on deployed model (public endpoint, uses API key)"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # TODO: Implement actual inference logic
    # This would call the deployed model with the image

    # Mock response for now
    request_id = str(uuid.uuid4())
    predictions = [
        PredictionResult(
            class_id=0,
            class_name="defect",
            confidence=0.85,
            bbox={"x": 100, "y": 150, "width": 200, "height": 180}
        )
    ]

    return InferenceResponse(
        request_id=request_id,
        predictions=predictions,
        inference_time=23.5,
        preprocessing_time=5.2,
        postprocessing_time=2.1
    )


@router.post("/{deployment_id}/health-check")
async def health_check(
    deployment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Perform health check on deployment"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # TODO: Implement actual health check
    # Ping the deployment endpoint

    db.commit()

    return {
        "deployment_id": deployment_id,
        "health_status": "healthy",
        "last_check": datetime.utcnow()
    }


@router.post("/{deployment_id}/start")
async def start_deployment(
    deployment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Start a deployment"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.deployed_at = datetime.utcnow()

    db.commit()

    return {"message": "Deployment started", "deployed_at": deployment.deployed_at}


@router.post("/{deployment_id}/stop")
async def stop_deployment(
    deployment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Stop a deployment"""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # status 필드가 없으므로 단순히 응답만 반환
    db.commit()

    return {"message": "Deployment stopped"}
