"""
AI Training Service
Provides a unified interface for model training across different frameworks.
"""

from pathlib import Path
from typing import Dict, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class YOLOTrainingService:
    """YOLO model training service using the AI module"""
    
    def __init__(self, model_spec: str = "yolov8n.pt"):
        """
        Initialize YOLO training service
        
        Args:
            model_spec: Base model to start from (e.g., "yolov8n.pt", "yolov8s.pt")
        """
        self.model_spec = model_spec
        self.adapter = None
        
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        name: str = "yolo_training",
        project: Optional[str] = None,
        exist_ok: bool = True,
        progress_callback: Optional[Callable] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            name: Training run name
            project: Project directory (default: runs/detect)
            exist_ok: Allow overwriting existing run
            progress_callback: Callback function for progress updates
            device: Device to use (0 for GPU, 'cpu' for CPU, None for auto)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        try:
            # Fix PyTorch 2.6 weights_only issue
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                # PyTorch 2.6+ - add Ultralytics classes to safe globals
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    torch.serialization.add_safe_globals([DetectionModel])
                except Exception as e:
                    logger.warning(f"Could not add safe globals: {e}")
            
            from model_trainer.integrations.yolo import build_yolo_model
            
            logger.info(f"Starting YOLO training with {self.model_spec}")
            logger.info(f"  Dataset: {data_yaml}")
            logger.info(f"  Epochs: {epochs}")
            logger.info(f"  Image size: {imgsz}")
            logger.info(f"  Batch size: {batch}")
            
            # Build model adapter
            self.adapter = build_yolo_model(self.model_spec)
            
            # Prepare training parameters
            train_params = {
                'data': data_yaml,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'name': name,
                'exist_ok': exist_ok,
            }
            
            if project:
                train_params['project'] = project
            
            if progress_callback:
                logger.info("✅ Progress callback provided - will track training progress")
                train_params['progress_callback'] = progress_callback
                train_params['tick_interval'] = 1.0  # Progress updates every second
            else:
                logger.warning("⚠️ No progress callback provided")
            
            # Add device if specified
            if device is not None:
                train_params['device'] = device
            
            # Merge additional kwargs
            train_params.update(kwargs)
            
            # Train using the adapter's fit method
            logger.info("Starting training...")
            result = self.adapter.fit(**train_params)
            
            # Extract results
            training_results = {
                'success': True,
                'model_spec': self.model_spec,
                'final_metrics': self.adapter.final_metrics or {},
                'save_dir': self.adapter.save_dir,
                'epochs_completed': self.adapter.last_epoch or epochs,
                'total_epochs': epochs
            }
            
            logger.info("✅ Training completed successfully")
            logger.info(f"  Model saved to: {self.adapter.save_dir}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_spec': self.model_spec
            }
    
    def get_best_model_path(self) -> Optional[Path]:
        """
        Get path to best trained model
        
        Returns:
            Path to best.pt file or None
        """
        if not self.adapter or not self.adapter.save_dir:
            return None
        
        best_path = Path(self.adapter.save_dir) / "weights" / "best.pt"
        if best_path.exists():
            return best_path
        
        return None


class TrainingServiceFactory:
    """Factory for creating training services"""
    
    @staticmethod
    def create_yolo_service(model_spec: str = "yolov8n.pt") -> YOLOTrainingService:
        """Create a YOLO training service"""
        return YOLOTrainingService(model_spec)


# Convenience function for quick training
def train_yolo(
    data_yaml: str,
    model_spec: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick YOLO training
    
    Args:
        data_yaml: Path to dataset YAML file
        model_spec: Base model (default: yolov8n.pt)
        epochs: Number of epochs
        imgsz: Image size
        batch: Batch size
        **kwargs: Additional parameters
        
    Returns:
        Training results dictionary
    """
    service = YOLOTrainingService(model_spec)
    return service.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("YOLO Training Service Example")
    print("=" * 60)
    print("\nTo use this service:")
    print("1. Prepare your dataset in YOLO format")
    print("2. Create a data.yaml file")
    print("3. Call train() with appropriate parameters")
    print("\nExample:")
    print("  service = YOLOTrainingService('yolov8n.pt')")
    print("  results = service.train(")
    print("      data_yaml='path/to/data.yaml',")
    print("      epochs=100,")
    print("      imgsz=640,")
    print("      batch=16")
    print("  )")

