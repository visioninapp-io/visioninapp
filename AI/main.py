"""
AI Service - FastAPI Application for GPU Server
Handles training and inference requests with GPU acceleration
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import uuid
import logging
import asyncio
from datetime import datetime
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI services
from training_service import YOLOTrainingService
from inference_service import YOLOInferenceService

app = FastAPI(
    title="AI Training & Inference Service",
    description="GPU-accelerated training and inference for YOLO models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (use Redis in production)
training_jobs: Dict[str, Dict[str, Any]] = {}
inference_cache: Dict[str, YOLOInferenceService] = {}

# ============================================================================
# Request/Response Models
# ============================================================================

class TrainingRequest(BaseModel):
    """Training job request"""
    job_id: str = Field(..., description="Unique job ID from backend")
    data_yaml: str = Field(..., description="Path to dataset YAML file")
    model: str = Field(default="yolov8n", description="Base YOLO model (yolov8n, yolov8s, etc.)")
    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=320, le=1280)
    batch: int = Field(default=16, ge=1, le=128)
    device: Optional[str] = Field(default=None, description="Device (auto-detect if None)")
    project: Optional[str] = Field(default=None, description="Project directory")
    name: Optional[str] = Field(default=None, description="Run name")
    
class TrainingResponse(BaseModel):
    """Training job response"""
    job_id: str
    status: str
    message: str
    
class TrainingStatusResponse(BaseModel):
    """Training status response"""
    job_id: str
    status: str
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    current_map50: float = 0.0
    metrics: Dict[str, Any] = {}
    error: Optional[str] = None
    
class InferenceRequest(BaseModel):
    """Inference request"""
    model_path: str = Field(..., description="Path to model .pt file")
    image_path: str = Field(..., description="Path to image file")
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    
class Detection(BaseModel):
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, float]  # x_center, y_center, width, height (normalized)
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] absolute
    
class InferenceResponse(BaseModel):
    """Inference response"""
    predictions: List[Detection]
    count: int
    inference_time: float

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Check service health and GPU availability"""
    cuda_available = torch.cuda.is_available()
    
    health_info = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    
    if cuda_available:
        health_info["device_name"] = torch.cuda.get_device_name(0)
        health_info["cuda_version"] = torch.version.cuda
        health_info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        health_info["device_name"] = "CPU"
        logger.warning("⚠️  CUDA not available - training will be slow!")
    
    return health_info

# ============================================================================
# Training Endpoints
# ============================================================================

def run_training_job(job_id: str, request: TrainingRequest):
    """Background task for training"""
    try:
        logger.info(f"[Job {job_id}] Starting training...")
        
        # Update status
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        
        # Determine device
        device = request.device
        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        
        # Initialize training service
        model_path = request.model if request.model.endswith('.pt') else f"{request.model}.pt"
        service = YOLOTrainingService(model_path)
        
        # Progress callback
        def progress_callback(status: Dict[str, Any]):
            """Update job status with training progress"""
            event = status.get('event', '')
            
            if event == 'epoch_end':
                epoch = status.get('epoch', 0)  # Already 1-based from callback
                total = status.get('total_epochs', request.epochs)
                metrics = status.get('metrics', {})
                
                # Log raw metrics to see what's available
                logger.info(f"[Job {job_id}] Raw metrics from epoch {epoch}: {list(metrics.keys())}")
                
                # Extract loss - try direct train_loss first, then sum components
                total_loss = metrics.get('train_loss', 0.0)
                if total_loss == 0.0:
                    # Fallback: sum loss components
                    box_loss = metrics.get('box_loss', 0.0)
                    cls_loss = metrics.get('cls_loss', 0.0)
                    dfl_loss = metrics.get('dfl_loss', 0.0)
                    total_loss = box_loss + cls_loss + dfl_loss
                
                # Extract mAP50 (try different possible key names)
                map50 = (metrics.get('metrics/mAP50(B)') or 
                        metrics.get('mAP50') or 
                        metrics.get('mAP50(B)') or 
                        metrics.get('val/mAP50') or 0.0)
                
                # Calculate progress
                progress = (epoch / total) * 100 if total > 0 else 0.0
                
                # Update job status
                training_jobs[job_id].update({
                    "status": "running",
                    "progress": progress,
                    "current_epoch": epoch,
                    "total_epochs": total,
                    "current_loss": total_loss,
                    "current_map50": map50,
                    "metrics": metrics
                })
                
                logger.info(f"[Job {job_id}] Epoch {epoch}/{total} ({progress:.1f}%) - Loss: {total_loss:.4f}, mAP50: {map50:.3f}")
        
        # Train
        result = service.train(
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            device=device,
            project=request.project,
            name=request.name or f"job_{job_id}",
            exist_ok=True,
            progress_callback=progress_callback
        )
        
        # Update final status
        if result['success']:
            training_jobs[job_id].update({
                "status": "completed",
                "progress": 100.0,
                "completed_at": datetime.utcnow().isoformat(),
                "result": {
                    "save_dir": str(result.get('save_dir', '')),
                    "best_model": str(result.get('best_model_path', '')),
                    "final_metrics": result.get('final_metrics', {})
                }
            })
            logger.info(f"[Job {job_id}] ✅ Training completed successfully")
        else:
            raise Exception(result.get('error', 'Unknown error'))
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Job {job_id}] ❌ Training failed: {error_msg}")
        
        # Provide user-friendly error messages
        if "weights_only" in error_msg or "WeightsUnpickler" in error_msg:
            error_msg = "PyTorch compatibility issue. Please downgrade PyTorch to 2.1.0 or install compatible Ultralytics version."
        
        training_jobs[job_id].update({
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.utcnow().isoformat(),
            "progress": 0.0
        })
        
        import traceback
        logger.error(f"[Job {job_id}] Traceback: {traceback.format_exc()}")

@app.post("/api/v1/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training job"""
    job_id = request.job_id
    
    # Check if job already exists
    if job_id in training_jobs:
        current_status = training_jobs[job_id]["status"]
        if current_status == "running":
            raise HTTPException(status_code=409, detail=f"Job {job_id} is already running")
    
    # Initialize job tracking
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "current_loss": 0.0,
        "current_map50": 0.0,
        "metrics": {},
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, request)
    
    logger.info(f"[Job {job_id}] Training job queued")
    
    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training job started"
    )

@app.get("/api/v1/train/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = training_jobs[job_id]
    
    return TrainingStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        current_epoch=job.get("current_epoch", 0),
        total_epochs=job.get("total_epochs", 0),
        current_loss=job.get("current_loss", 0.0),
        current_map50=job.get("current_map50", 0.0),
        metrics=job.get("metrics", {}),
        error=job.get("error")
    )

@app.delete("/api/v1/train/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = training_jobs[job_id]
    if job["status"] == "running":
        # TODO: Implement actual cancellation (requires threading/multiprocessing control)
        job["status"] = "cancelled"
        job["completed_at"] = datetime.utcnow().isoformat()
        return {"message": f"Job {job_id} cancelled"}
    
    return {"message": f"Job {job_id} is not running (status: {job['status']})"}

# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/api/v1/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Run inference on an image"""
    import time
    
    try:
        # Check if model exists
        model_path = Path(request.model_path)
        if not model_path.exists():
            # Try default models directory
            model_path = Path("models") / model_path.name
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        # Check if image exists
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        # Load model (use cache)
        model_key = str(model_path)
        if model_key not in inference_cache:
            logger.info(f"Loading model: {model_path}")
            service = YOLOInferenceService(str(model_path))
            if not service.load_model():
                raise HTTPException(status_code=500, detail="Failed to load model")
            inference_cache[model_key] = service
        else:
            service = inference_cache[model_key]
        
        # Run inference
        start_time = time.time()
        detections = service.predict(
            source=str(image_path),
            conf=request.conf,
            iou=request.iou
        )
        inference_time = time.time() - start_time
        
        # Format response
        predictions = [Detection(**det) for det in detections]
        
        logger.info(f"Inference completed: {len(predictions)} detections in {inference_time:.3f}s")
        
        return InferenceResponse(
            predictions=predictions,
            count=len(predictions),
            inference_time=inference_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/v1/predict/batch")
async def predict_batch(model_path: str, image_paths: List[str], conf: float = 0.25):
    """Run inference on multiple images"""
    results = []
    
    for image_path in image_paths:
        try:
            result = await predict(InferenceRequest(
                model_path=model_path,
                image_path=image_path,
                conf=conf
            ))
            results.append({
                "image_path": image_path,
                "predictions": result.predictions,
                "success": True
            })
        except Exception as e:
            results.append({
                "image_path": image_path,
                "error": str(e),
                "success": False
            })
    
    return {"results": results}

# ============================================================================
# Model Management
# ============================================================================

@app.get("/api/v1/models")
async def list_models():
    """List available models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for model_file in models_dir.glob("*.pt"):
        models.append({
            "name": model_file.name,
            "path": str(model_file),
            "size_mb": model_file.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
        })
    
    return {"models": models}

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("=" * 60)
    logger.info("AI Service Starting...")
    logger.info("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA Available")
        logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA Not Available - Using CPU")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("AI Service Ready!")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("AI Service shutting down...")
    
    # Clear inference cache
    inference_cache.clear()
    
    # Report active jobs
    active_jobs = [j for j in training_jobs.values() if j["status"] == "running"]
    if active_jobs:
        logger.warning(f"⚠️  {len(active_jobs)} training jobs were still running")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

