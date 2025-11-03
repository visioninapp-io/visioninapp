"""
AI Service Client
HTTP client to communicate with GPU AI service
"""

import httpx
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AIServiceClient:
    """Client to communicate with AI service on GPU server"""
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 300.0):
        """
        Initialize AI service client
        
        Args:
            base_url: AI service URL (default: from AI_SERVICE_URL env var or localhost)
            timeout: Request timeout in seconds (default: 5 minutes for training)
        """
        self.base_url = base_url or os.getenv("AI_SERVICE_URL", "http://localhost:8001")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"AI Service Client initialized: {self.base_url}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check AI service health and GPU availability
        
        Returns:
            Health status dict with CUDA info
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            
            cuda_status = "✅ GPU" if health.get("cuda_available") else "⚠️  CPU"
            logger.info(f"AI Service Health: {cuda_status} - {health.get('device_name', 'Unknown')}")
            
            return health
        except Exception as e:
            logger.error(f"AI Service health check failed: {e}")
            raise Exception(f"AI Service unavailable: {e}")
    
    async def start_training(
        self,
        job_id: int,
        data_yaml: str,
        model: str = "yolov8n",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start training job on AI service
        
        Args:
            job_id: Training job ID from database
            data_yaml: Path to dataset YAML file
            model: Base YOLO model (yolov8n, yolov8s, etc.)
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            device: Device to use (None for auto)
            project: Project directory
            name: Run name
            
        Returns:
            Training job response
        """
        try:
            payload = {
                "job_id": str(job_id),
                "data_yaml": str(data_yaml),
                "model": model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch
            }
            
            if device:
                payload["device"] = device
            if project:
                payload["project"] = str(project)
            if name:
                payload["name"] = name
            
            logger.info(f"[Job {job_id}] Starting training on AI service...")
            logger.info(f"  Model: {model}, Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/train",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"[Job {job_id}] Training started: {result.get('status')}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[Job {job_id}] Training request failed: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Training request failed: {e.response.text}")
        except Exception as e:
            logger.error(f"[Job {job_id}] Failed to start training: {e}")
            raise
    
    async def get_training_status(self, job_id: int) -> Dict[str, Any]:
        """
        Get training job status from AI service
        
        Args:
            job_id: Training job ID
            
        Returns:
            Training status dict with progress, metrics, etc.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/train/{job_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"[Job {job_id}] Not found on AI service")
                return {"status": "unknown", "error": "Job not found on AI service"}
            raise
        except Exception as e:
            logger.error(f"[Job {job_id}] Failed to get training status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cancel_training(self, job_id: int) -> Dict[str, Any]:
        """
        Cancel training job on AI service
        
        Args:
            job_id: Training job ID
            
        Returns:
            Cancellation result
        """
        try:
            response = await self.client.delete(f"{self.base_url}/api/v1/train/{job_id}")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"[Job {job_id}] Training cancelled")
            
            return result
        except Exception as e:
            logger.error(f"[Job {job_id}] Failed to cancel training: {e}")
            raise
    
    async def predict(
        self,
        model_path: str,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> Dict[str, Any]:
        """
        Run inference on AI service
        
        Args:
            model_path: Path to model .pt file
            image_path: Path to image file
            conf: Confidence threshold
            iou: IOU threshold for NMS
            
        Returns:
            Inference results with predictions
        """
        try:
            payload = {
                "model_path": str(model_path),
                "image_path": str(image_path),
                "conf": conf,
                "iou": iou
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/predict",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Inference: {result['count']} detections in {result['inference_time']:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    async def predict_batch(
        self,
        model_path: str,
        image_paths: List[str],
        conf: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run batch inference on AI service
        
        Args:
            model_path: Path to model .pt file
            image_paths: List of image paths
            conf: Confidence threshold
            
        Returns:
            Batch inference results
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/predict/batch",
                params={
                    "model_path": str(model_path),
                    "conf": conf
                },
                json=image_paths
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on AI service
        
        Returns:
            List of model info dicts
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance
_ai_client: Optional[AIServiceClient] = None


def get_ai_client() -> AIServiceClient:
    """
    Get AI service client singleton
    
    Returns:
        AIServiceClient instance
    """
    global _ai_client
    if _ai_client is None:
        _ai_client = AIServiceClient()
    return _ai_client

