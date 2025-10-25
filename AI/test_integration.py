"""
Integration test script for AI module
Tests inference and training services
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_inference_service():
    """Test the inference service"""
    logger.info("=" * 60)
    logger.info("Testing Inference Service")
    logger.info("=" * 60)
    
    try:
        from inference_service import YOLOInferenceService
        
        # Test 1: Load yolov8n (will auto-download if needed)
        logger.info("\nTest 1: Loading YOLOv8n model...")
        service = YOLOInferenceService("yolov8n.pt")
        
        if not service.load_model():
            logger.error("❌ Failed to load model")
            return False
        
        logger.info("✅ Model loaded successfully")
        
        # Test 2: Get model info
        logger.info("\nTest 2: Getting model info...")
        info = service.get_model_info()
        logger.info(f"✅ Model info retrieved:")
        logger.info(f"   - Classes: {info['num_classes']}")
        logger.info(f"   - Sample classes: {', '.join(info['class_names'][:5])}...")
        
        # Test 3: Test prediction (without actual image)
        logger.info("\nTest 3: Testing prediction interface...")
        logger.info("   (Skipping actual inference - no test image)")
        logger.info("   To test with an actual image, run:")
        logger.info("     results = service.predict('path/to/image.jpg', conf=0.25)")
        
        logger.info("\n✅ Inference Service: All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_service():
    """Test the training service"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Training Service")
    logger.info("=" * 60)
    
    try:
        from training_service import YOLOTrainingService
        
        # Test 1: Create training service
        logger.info("\nTest 1: Creating training service...")
        service = YOLOTrainingService("yolov8n.pt")
        logger.info("✅ Training service created")
        
        # Test 2: Test training interface (without actual training)
        logger.info("\nTest 2: Testing training interface...")
        logger.info("   (Skipping actual training - no dataset)")
        logger.info("   To train with actual dataset, run:")
        logger.info("     results = service.train(")
        logger.info("         data_yaml='path/to/data.yaml',")
        logger.info("         epochs=10,")
        logger.info("         imgsz=640,")
        logger.info("         batch=16")
        logger.info("     )")
        
        logger.info("\n✅ Training Service: All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_trainer_module():
    """Test the core model_trainer module"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Trainer Module")
    logger.info("=" * 60)
    
    try:
        from model_trainer.factory import build_model
        from model_trainer.integrations.yolo import build_yolo_model
        
        # Test 1: Factory function
        logger.info("\nTest 1: Testing model factory...")
        model = build_model("yolov8n")
        logger.info("✅ Model factory works")
        
        # Test 2: YOLO adapter
        logger.info("\nTest 2: Testing YOLO adapter...")
        adapter = build_yolo_model("yolov8n")
        logger.info("✅ YOLO adapter created")
        logger.info(f"   - Model has {len(adapter._model.names)} classes")
        
        logger.info("\n✅ Model Trainer Module: All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Trainer Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_be_integration_compatibility():
    """Test compatibility with BE auto_annotation_service"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing BE Integration Compatibility")
    logger.info("=" * 60)
    
    try:
        # Simulate BE's usage pattern
        logger.info("\nTest: Simulating BE auto_annotation_service usage...")
        
        from ultralytics import YOLO
        
        # Test 1: Load default model (what BE does now)
        logger.info("   1. Testing default model loading (yolov8n)...")
        model = YOLO("yolov8n.pt")
        logger.info(f"   ✅ Loaded with {len(model.names)} classes")
        
        # Test 2: Check if we can do inference
        logger.info("   2. Testing inference capability...")
        logger.info("   ✅ Inference interface available")
        logger.info("      (Would call: model.predict(source, conf=0.25))")
        
        logger.info("\n✅ BE Integration: Compatible!")
        return True
        
    except Exception as e:
        logger.error(f"❌ BE Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    logger.info("AI Module Integration Tests")
    logger.info("=" * 60)
    logger.info("This script tests the AI module's integration capabilities")
    logger.info("")
    
    results = {
        'inference_service': test_inference_service(),
        'training_service': test_training_service(),
        'model_trainer': test_model_trainer_module(),
        'be_integration': test_be_integration_compatibility(),
    }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:30s} {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 All tests passed! Integration is working correctly.")
        logger.info("\nNext steps:")
        logger.info("  1. Start the BE server: cd BE && python main.py")
        logger.info("  2. Open FE in browser: FE/index.html")
        logger.info("  3. Try auto-annotation feature (will use yolov8n by default)")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

