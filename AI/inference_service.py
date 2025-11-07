"""
YOLO Inference Service
High-level interface for YOLO model inference and auto-annotation
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class YOLOInferenceService:
    """High-level YOLO inference service for auto-annotation"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference service
        
        Args:
            model_path: Path to model file (optional, will use default if not provided)
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self.is_loaded = False
        
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Inference device: {self.device}")
        
        # Auto-load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load YOLO model
        
        Args:
            model_path: Path to model file (optional)
        """
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            # Use default YOLOv8n model
            self.model_path = "yolov8n.pt"
            logger.info("No model specified, using default YOLOv8n")
        
        try:
            from ultralytics import YOLO
            
            # Check if model file exists
            model_path = Path(self.model_path)
            if not model_path.exists() and not self.model_path.startswith('yolo'):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"📦 Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move to device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.model_path}")
            
            # Log model info
            if hasattr(self.model, 'names') and self.model.names:
                logger.info(f" Model classes: {len(self.model.names)} ({list(self.model.names.values())[:5]}...)")
            
        except Exception as e:
            logger.error(f" Failed to load model {self.model_path}: {e}")
            self.is_loaded = False
            raise RuntimeError(f"Model loading failed: {e}")
    
    def predict(self, 
                image_path: Union[str, Path, Image.Image, np.ndarray],
                conf: float = 0.25,
                iou: float = 0.45,
                imgsz: int = 640,
                max_det: int = 300,
                classes: Optional[List[int]] = None,
                verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Run inference on image(s)
        
        Args:
            image_path: Path to image file, PIL Image, or numpy array
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            max_det: Maximum detections per image
            classes: List of class IDs to detect (None for all)
            verbose: Verbose output
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info(f" Running inference on: {image_path}")
            logger.info(f"   Confidence: {conf}, IoU: {iou}, Image size: {imgsz}")
            
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                classes=classes,
                verbose=verbose,
                save=False,
                show=False
            )
            
            # Process results
            detections = []
            for result in results:
                detections.extend(self._process_result(result))
            
            logger.info(f" Inference completed: {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f" Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
    
    def _process_result(self, result) -> List[Dict[str, Any]]:
        """Process YOLO result into standardized format"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Get image dimensions
        img_height, img_width = result.orig_shape
        
        # Process each detection
        boxes = result.boxes
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # Convert to normalized center format (YOLO format)
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            
            # Get class name
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': {
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height)
                },
                'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'image_size': {
                    'width': img_width,
                    'height': img_height
                }
            }
            
            detections.append(detection)
        
        return detections
    
    def predict_batch(self, 
                     image_paths: List[Union[str, Path]],
                     conf: float = 0.25,
                     iou: float = 0.45,
                     imgsz: int = 640,
                     max_det: int = 300,
                     classes: Optional[List[int]] = None,
                     verbose: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            max_det: Maximum detections per image
            classes: List of class IDs to detect (None for all)
            verbose: Verbose output
            
        Returns:
            Dictionary mapping image paths to detection lists
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = {}
        
        logger.info(f" Running batch inference on {len(image_paths)} images")
        
        for image_path in image_paths:
            try:
                detections = self.predict(
                    image_path=image_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    classes=classes,
                    verbose=verbose
                )
                results[str(image_path)] = detections
                
            except Exception as e:
                logger.error(f" Failed to process {image_path}: {e}")
                results[str(image_path)] = []
        
        total_detections = sum(len(dets) for dets in results.values())
        logger.info(f" Batch inference completed: {total_detections} total detections")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded
        }
        
        # Add model-specific info
        try:
            if hasattr(self.model, 'names') and self.model.names:
                info["classes"] = len(self.model.names)
                info["class_names"] = list(self.model.names.values())
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
                info["model_yaml"] = self.model.model.yaml
                
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
        
        return info
    
    def annotate_for_backend(self, 
                           image_path: Union[str, Path],
                           conf: float = 0.25,
                           iou: float = 0.45) -> Dict[str, Any]:
        """
        Run inference and format results for backend auto-annotation
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            Formatted annotation data for backend
        """
        try:
            detections = self.predict(
                image_path=image_path,
                conf=conf,
                iou=iou
            )
            
            # Format for backend
            annotations = []
            for det in detections:
                annotation = {
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "x_center": det["bbox"]["x_center"],
                    "y_center": det["bbox"]["y_center"],
                    "width": det["bbox"]["width"],
                    "height": det["bbox"]["height"]
                }
                annotations.append(annotation)
            
            return {
                "success": True,
                "image_path": str(image_path),
                "annotations": annotations,
                "total_detections": len(annotations),
                "model_info": {
                    "model_path": self.model_path,
                    "confidence_threshold": conf,
                    "iou_threshold": iou
                }
            }
            
        except Exception as e:
            logger.error(f" Auto-annotation failed for {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path),
                "annotations": []
            }


# Convenience functions
def auto_annotate_image(image_path: Union[str, Path], 
                       model_path: Optional[str] = None,
                       conf: float = 0.25,
                       iou: float = 0.45) -> Dict[str, Any]:
    """
    Convenience function for auto-annotation
    
    Args:
        image_path: Path to image
        model_path: Path to model (optional, uses default)
        conf: Confidence threshold
        iou: IoU threshold
        
    Returns:
        Annotation results
    """
    service = YOLOInferenceService(model_path)
    return service.annotate_for_backend(image_path, conf, iou)

def batch_auto_annotate(image_paths: List[Union[str, Path]], 
                       model_path: Optional[str] = None,
                       conf: float = 0.25,
                       iou: float = 0.45) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function for batch auto-annotation
    
    Args:
        image_paths: List of image paths
        model_path: Path to model (optional, uses default)
        conf: Confidence threshold
        iou: IoU threshold
        
    Returns:
        Dictionary mapping image paths to annotation results
    """
    service = YOLOInferenceService(model_path)
    results = {}
    
    for image_path in image_paths:
        results[str(image_path)] = service.annotate_for_backend(image_path, conf, iou)
    
    return results
