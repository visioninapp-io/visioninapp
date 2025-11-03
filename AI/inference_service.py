"""
AI Inference Service
Provides a unified interface for model inference across different frameworks.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class YOLOInferenceService:
    """YOLO model inference service using ultralytics"""
    
    def __init__(self, model_spec: Union[str, Path] = "yolov8n.pt"):
        """
        Initialize YOLO inference service
        
        Args:
            model_spec: Model specification - can be:
                - "yolov8n.pt", "yolov8s.pt", etc. (will auto-download)
                - Path to custom .pt file
                - "yolov8n" (will add .pt automatically)
        """
        from model_trainer.integrations.yolo import YOLOAdapter
        
        self.model_spec = str(model_spec)
        self.adapter = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the YOLO model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from model_trainer.integrations.yolo import build_yolo_model
            
            logger.info(f"Loading YOLO model: {self.model_spec}")
            self.adapter = build_yolo_model(self.model_spec)
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Model: {self.model_spec}")
            logger.info(f"   Classes: {len(self.adapter._model.names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(
        self,
        source: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image(s)
        
        Args:
            source: Image path or directory
            conf: Confidence threshold
            iou: IOU threshold for NMS
            **kwargs: Additional inference parameters
            
        Returns:
            List of detections with format:
            [{
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': {
                    'x_center': float,  # normalized 0-1
                    'y_center': float,
                    'width': float,
                    'height': float
                },
                'bbox_xyxy': [x1, y1, x2, y2]  # absolute coordinates
            }, ...]
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        try:
            results = self.adapter.predict(source, conf=conf, iou=iou, **kwargs)
            
            if not results or len(results) == 0:
                return []
            
            # Parse results
            annotations = []
            result = results[0]
            
            if not hasattr(result, 'boxes') or result.boxes is None:
                return []
            
            boxes = result.boxes
            img_height, img_width = result.orig_shape
            
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                cls_name = self.adapter._model.names[cls_id]
                
                # XYXY coordinates (absolute)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # YOLO format (normalized center + width/height)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                annotation = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': confidence,
                    'bbox': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    },
                    'bbox_xyxy': [x1, y1, x2, y2]
                }
                
                annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded:
            return {
                'model_spec': self.model_spec,
                'is_loaded': False,
                'num_classes': 0,
                'class_names': [],
            }
        
        return {
            'model_spec': self.model_spec,
            'is_loaded': True,
            'num_classes': len(self.adapter._model.names),
            'class_names': list(self.adapter._model.names.values()),
        }


class InferenceServiceFactory:
    """Factory for creating inference services"""
    
    @staticmethod
    def create_yolo_service(model_spec: Union[str, Path] = "yolov8n.pt") -> YOLOInferenceService:
        """Create a YOLO inference service"""
        return YOLOInferenceService(model_spec)
    
    @staticmethod
    def auto_detect_and_create(model_path: Union[str, Path]) -> Optional[YOLOInferenceService]:
        """
        Auto-detect model type and create appropriate service
        
        Args:
            model_path: Path to model file
            
        Returns:
            Inference service instance or None
        """
        model_str = str(model_path).lower()
        
        # Check if it's a YOLO model
        if 'yolo' in model_str or model_str.endswith('.pt'):
            return YOLOInferenceService(model_path)
        
        logger.error(f"Could not detect model type for: {model_path}")
        return None


# Convenience function for quick inference
def predict_with_yolo(
    image_path: Union[str, Path],
    model_spec: Union[str, Path] = "yolov8n.pt",
    conf: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Quick inference with YOLO
    
    Args:
        image_path: Path to image
        model_spec: Model specification (default: yolov8n.pt)
        conf: Confidence threshold
        
    Returns:
        List of detections
    """
    service = YOLOInferenceService(model_spec)
    return service.predict(image_path, conf=conf)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with default YOLOv8n model
    service = YOLOInferenceService("yolov8n.pt")
    service.load_model()
    
    info = service.get_model_info()
    print(f"Model Info: {info}")
    
    # Example inference (uncomment and provide real image path)
    # results = service.predict("path/to/image.jpg", conf=0.25)
    # print(f"Detections: {results}")

