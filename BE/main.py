from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.core.config import settings
from app.core.database import engine, Base
from app.core.auth import init_firebase
from app.api.v1.api import api_router
import os

# Import all models to ensure they are registered with Base
import app.models

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize Firebase
init_firebase()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="""
    VisionAI Platform API - Complete AI Object Detection Lifecycle Management

    ## Features

    * **Datasets** - Upload, annotate, and manage training datasets
    * **Training** - Train models with hyperparameter tuning
    * **Models** - Model conversion and optimization (ONNX, TensorRT)
    * **Evaluation** - Comprehensive performance metrics and comparison
    * **Deployment** - Deploy to edge devices, cloud, and on-premise
    * **Monitoring** - Real-time monitoring and feedback loops

    ## Authentication

    All endpoints (except health check) require Firebase authentication.
    Include the Firebase ID token in the Authorization header:

    ```
    Authorization: Bearer <firebase_id_token>
    ```
    """
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to VisionAI Platform API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION
    }

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files for uploaded images
upload_dir = Path(settings.UPLOAD_DIR)
upload_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")

# Mount datasets directory for image access
datasets_dir = upload_dir / "datasets"
datasets_dir.mkdir(exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")


# ========== RabbitMQ Consumer 시작 ==========
@app.on_event("startup")
def start_rabbitmq_consumers():
    """Start RabbitMQ consumers in background threads"""
    from app.rabbitmq.consumer import start_inference_consumer
    from app.services.auto_annotation_service import handle_inference_done
    import logging
    import threading

    log = logging.getLogger(__name__)

    def run_consumer():
        try:
            log.info("[RMQ] Starting inference.done consumer...")
            start_inference_consumer(handle_inference_done)
        except Exception as e:
            log.error(f"[RMQ] Consumer error: {e}", exc_info=True)

    # Background thread로 Consumer 시작
    consumer_thread = threading.Thread(target=run_consumer, daemon=True)
    consumer_thread.start()
    log.info("[RMQ] Consumer thread started")
