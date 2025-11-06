"""
Model Conversion Example
Demonstrates how to convert YOLO models to ONNX and TensorRT formats
"""

import sys
from pathlib import Path

# Add AI module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conversion_service import ModelConversionService, convert_to_onnx, convert_to_tensorrt, get_model_info
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate model conversion functionality"""
    
    print("=" * 60)
    print("YOLO Model Conversion Example")
    print("=" * 60)
    
    # Initialize conversion service
    service = ModelConversionService()
    
    # Example model path (you can change this to your model)
    model_path = "yolov8n.pt"  # Will auto-download if not present
    
    print(f"\n1. Getting model information...")
    print("-" * 40)
    
    # Get model information
    info = get_model_info(model_path)
    if "error" not in info:
        print(f"Model: {info['model_path']}")
        print(f"Size: {info['model_size_mb']:.1f} MB")
        print(f"Classes: {info['classes']}")
        print(f"Class names: {info['class_names'][:5]}...")  # Show first 5 classes
        print(f"Supported formats: {info['supported_formats']}")
    else:
        print(f"Error getting model info: {info['error']}")
        return
    
    print(f"\n2. Converting to ONNX...")
    print("-" * 40)
    
    # Convert to ONNX with default settings
    onnx_result = convert_to_onnx(
        model_path=model_path,
        output_dir="converted_models",
        imgsz=640,
        batch_size=1,
        dynamic=False,
        simplify=True,
        opset=17
    )
    
    if onnx_result["success"]:
        print(f"✅ ONNX conversion successful!")
        print(f"   Output: {onnx_result['output_path']}")
        print(f"   Time: {onnx_result['conversion_time']:.2f}s")
        print(f"   Size: {onnx_result['model_info']['source_size_mb']:.1f}MB -> {onnx_result['model_info']['output_size_mb']:.1f}MB")
    else:
        print(f"❌ ONNX conversion failed: {onnx_result['error']}")
    
    print(f"\n3. Converting to ONNX with dynamic shapes...")
    print("-" * 40)
    
    # Convert to ONNX with dynamic shapes
    onnx_dynamic_result = convert_to_onnx(
        model_path=model_path,
        output_dir="converted_models",
        imgsz=640,
        batch_size=1,
        dynamic=True,  # Enable dynamic shapes
        simplify=True,
        opset=17
    )
    
    if onnx_dynamic_result["success"]:
        print(f"✅ Dynamic ONNX conversion successful!")
        print(f"   Output: {onnx_dynamic_result['output_path']}")
        print(f"   Time: {onnx_dynamic_result['conversion_time']:.2f}s")
    else:
        print(f"❌ Dynamic ONNX conversion failed: {onnx_dynamic_result['error']}")
    
    print(f"\n4. Converting to TensorRT (FP16)...")
    print("-" * 40)
    
    # Convert to TensorRT with FP16 precision
    try:
        tensorrt_result = convert_to_tensorrt(
            model_path=model_path,
            output_dir="converted_models",
            imgsz=640,
            batch_size=1,
            precision="fp16",
            workspace=4
        )
        
        if tensorrt_result["success"]:
            print(f"✅ TensorRT conversion successful!")
            print(f"   Output: {tensorrt_result['output_path']}")
            print(f"   Time: {tensorrt_result['conversion_time']:.2f}s")
            print(f"   Precision: {tensorrt_result['metadata']['precision']}")
        else:
            print(f"❌ TensorRT conversion failed: {tensorrt_result['error']}")
    except Exception as e:
        print(f"❌ TensorRT conversion failed: {e}")
        print("   Note: TensorRT requires NVIDIA GPU and TensorRT installation")
    
    print(f"\n5. Batch conversion example...")
    print("-" * 40)
    
    # Example of batch conversion (if you have multiple models)
    model_paths = [model_path]  # Add more model paths here
    
    batch_results = service.batch_convert(
        model_paths=model_paths,
        target_format="onnx",
        output_dir="converted_models",
        imgsz=416,  # Different size for demonstration
        batch_size=1,
        dynamic=False,
        simplify=True
    )
    
    successful = sum(1 for r in batch_results if r.get("success", False))
    print(f"Batch conversion: {successful}/{len(batch_results)} successful")
    
    print(f"\n6. Advanced conversion options...")
    print("-" * 40)
    
    # Show advanced conversion with custom options
    advanced_result = service.convert_model(
        model_path=model_path,
        target_format="onnx",
        output_dir="converted_models",
        imgsz=320,  # Smaller size for edge deployment
        batch_size=4,  # Larger batch
        dynamic=True,
        simplify=True,
        opset=16,  # Different opset version
    )
    
    if advanced_result["success"]:
        print(f"✅ Advanced ONNX conversion successful!")
        print(f"   Output: {advanced_result['output_path']}")
        print(f"   Settings: {advanced_result['conversion_options']}")
    else:
        print(f"❌ Advanced conversion failed: {advanced_result['error']}")
    
    print(f"\n" + "=" * 60)
    print("Conversion example completed!")
    print("=" * 60)
    
    # Show output directory contents
    output_dir = Path("converted_models")
    if output_dir.exists():
        print(f"\nConverted models in '{output_dir}':")
        for model_file in output_dir.glob("*"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
