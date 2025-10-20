from fastapi import APIRouter
from app.api.v1.endpoints import (
    datasets,
    training,
    models,
    evaluation,
    deployment,
    monitoring
)

api_router = APIRouter()

api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(deployment.router, prefix="/deployment", tags=["deployment"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
