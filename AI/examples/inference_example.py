"""
Example script demonstrating how to use the AI module for inference
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_service import YOLOInferenceService, predict_with_yolo
import logging

logging.basicConfig(level=logging.INFO)


def example_basic_inference():
    """Basic inference example using YOLOv8n"""
    print("=" * 60)
    print("Example 1: Basic Inference with YOLOv8n")
    print("=" * 60)
    
    # Create service with default YOLOv8n model (will auto-download)
    service = YOLOInferenceService("yolov8n.pt")
    
    # Load model
    if not service.load_model():
        print("Failed to load model")
        return
    
    # Get model info
    info = service.get_model_info()
    print(f"\nModel loaded successfully!")
    print(f"  - Classes: {info['num_classes']}")
    print(f"  - Available classes: {', '.join(info['class_names'][:10])}...")
    
    print("\nTo run inference, call:")
    print('  results = service.predict("path/to/image.jpg", conf=0.25)')


def example_custom_model():
    """Example using a custom trained model"""
    print("\n" + "=" * 60)
    print("Example 2: Using Custom Trained Model")
    print("=" * 60)
    
    # Path to custom model
    custom_model_path = Path(__file__).parent.parent / "models" / "best.pt"
    
    if not custom_model_path.exists():
        print(f"\nCustom model not found at: {custom_model_path}")
        print("Train a model first using train_yolo.py")
        print("\nFalling back to YOLOv8n for demonstration...")
        custom_model_path = "yolov8n.pt"
    
    # Create service with custom model
    service = YOLOInferenceService(custom_model_path)
    
    if service.load_model():
        info = service.get_model_info()
        print(f"\nModel loaded: {custom_model_path}")
        print(f"  - Classes: {info['num_classes']}")


def example_quick_predict():
    """Example using the convenience function"""
    print("\n" + "=" * 60)
    print("Example 3: Quick Prediction (Convenience Function)")
    print("=" * 60)
    
    print("\nFor quick one-off predictions, use:")
    print('  from inference_service import predict_with_yolo')
    print('  results = predict_with_yolo("image.jpg", "yolov8n.pt", conf=0.25)')
    print("\nThis automatically handles model loading and inference.")


def example_integration_with_be():
    """Example showing how BE should integrate with AI module"""
    print("\n" + "=" * 60)
    print("Example 4: Integration Pattern for Backend")
    print("=" * 60)
    
    print("""
Backend Integration Pattern:

1. Import the inference service:
   from AI.inference_service import YOLOInferenceService

2. Create and cache service instance:
   service = YOLOInferenceService("yolov8n.pt")
   service.load_model()

3. Use for predictions:
   annotations = service.predict(image_path, conf=0.25)

4. Handle results:
   for ann in annotations:
       class_name = ann['class_name']
       confidence = ann['confidence']
       bbox = ann['bbox']  # normalized coordinates
       # Save to database...

5. For custom models:
   # If user has trained model
   service = YOLOInferenceService("path/to/custom/model.pt")
   
   # If no custom model, fallback to yolov8n
   if not custom_model_exists:
       service = YOLOInferenceService("yolov8n.pt")
""")


def main():
    """Run all examples"""
    print("AI Module Inference Examples")
    print("=" * 60)
    
    example_basic_inference()
    example_custom_model()
    example_quick_predict()
    example_integration_with_be()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

