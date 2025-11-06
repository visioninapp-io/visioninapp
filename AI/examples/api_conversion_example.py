"""
API-based Model Conversion Example
Demonstrates how to use the FastAPI endpoints for model conversion
"""

import requests
import json
import time
from pathlib import Path


class ConversionAPIClient:
    """Client for interacting with the AI service conversion API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_health(self):
        """Check if the AI service is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Service not available. Make sure AI service is running on port 8001"}
    
    def get_supported_formats(self):
        """Get supported conversion formats and options"""
        response = self.session.get(f"{self.base_url}/api/v1/convert/formats")
        return response.json()
    
    def get_model_info(self, model_name: str):
        """Get information about a model"""
        response = self.session.get(f"{self.base_url}/api/v1/models/{model_name}/info")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    
    def convert_model(self, conversion_request: dict):
        """Convert a model using the API"""
        response = self.session.post(
            f"{self.base_url}/api/v1/convert",
            json=conversion_request
        )
        return response.json()
    
    def batch_convert(self, model_paths: list, target_format: str, **options):
        """Convert multiple models"""
        response = self.session.post(
            f"{self.base_url}/api/v1/convert/batch",
            json={
                "model_paths": model_paths,
                "target_format": target_format,
                "conversion_options": options
            }
        )
        return response.json()


def main():
    """Demonstrate API-based model conversion"""
    
    print("=" * 60)
    print("AI Service API Conversion Example")
    print("=" * 60)
    
    # Initialize API client
    client = ConversionAPIClient()
    
    print("\n1. Checking AI service health...")
    print("-" * 40)
    
    health = client.get_health()
    if "error" in health:
        print(f"❌ {health['error']}")
        print("\nTo start the AI service, run:")
        print("cd AI && python main.py")
        return
    
    print(f"✅ AI Service is running")
    print(f"   CUDA Available: {health.get('cuda_available', False)}")
    print(f"   Device: {health.get('device_name', 'Unknown')}")
    
    print("\n2. Getting supported formats...")
    print("-" * 40)
    
    formats = client.get_supported_formats()
    print(f"Supported formats: {formats['supported_formats']}")
    print(f"Supported precisions: {formats['supported_precisions']}")
    
    print("\n3. Getting model information...")
    print("-" * 40)
    
    # Get info for yolov8n model
    model_info = client.get_model_info("yolov8n.pt")
    if "error" not in model_info:
        print(f"Model: {model_info['model_path']}")
        print(f"Size: {model_info['model_size_mb']:.1f} MB")
        print(f"Classes: {model_info['classes']}")
        print(f"Recommended ONNX settings: {model_info['recommended_settings']['onnx']}")
    else:
        print(f"❌ Error: {model_info['error']}")
        print("Make sure yolov8n.pt is available in the models directory")
        return
    
    print("\n4. Converting to ONNX...")
    print("-" * 40)
    
    # Convert to ONNX
    onnx_request = {
        "model_path": "models/yolov8n.pt",
        "target_format": "onnx",
        "output_dir": "converted_models",
        "imgsz": 640,
        "batch_size": 1,
        "dynamic": False,
        "simplify": True,
        "opset": 17
    }
    
    onnx_result = client.convert_model(onnx_request)
    
    if onnx_result.get("success", False):
        print(f"✅ ONNX conversion successful!")
        print(f"   Output: {onnx_result['output_path']}")
        print(f"   Time: {onnx_result['conversion_time']:.2f}s")
        print(f"   Size change: {onnx_result['model_info']['source_size_mb']:.1f}MB -> {onnx_result['model_info']['output_size_mb']:.1f}MB")
    else:
        print(f"❌ ONNX conversion failed: {onnx_result.get('error', 'Unknown error')}")
    
    print("\n5. Converting to ONNX with dynamic shapes...")
    print("-" * 40)
    
    # Convert to ONNX with dynamic shapes
    dynamic_request = {
        "model_path": "models/yolov8n.pt",
        "target_format": "onnx",
        "output_dir": "converted_models",
        "imgsz": 640,
        "batch_size": 1,
        "dynamic": True,  # Enable dynamic shapes
        "simplify": True,
        "opset": 17
    }
    
    dynamic_result = client.convert_model(dynamic_request)
    
    if dynamic_result.get("success", False):
        print(f"✅ Dynamic ONNX conversion successful!")
        print(f"   Output: {dynamic_result['output_path']}")
        print(f"   Time: {dynamic_result['conversion_time']:.2f}s")
    else:
        print(f"❌ Dynamic ONNX conversion failed: {dynamic_result.get('error', 'Unknown error')}")
    
    print("\n6. Converting to TensorRT (if available)...")
    print("-" * 40)
    
    # Convert to TensorRT
    tensorrt_request = {
        "model_path": "models/yolov8n.pt",
        "target_format": "tensorrt",
        "output_dir": "converted_models",
        "imgsz": 640,
        "batch_size": 1,
        "precision": "fp16",
        "workspace": 4
    }
    
    tensorrt_result = client.convert_model(tensorrt_request)
    
    if tensorrt_result.get("success", False):
        print(f"✅ TensorRT conversion successful!")
        print(f"   Output: {tensorrt_result['output_path']}")
        print(f"   Time: {tensorrt_result['conversion_time']:.2f}s")
        print(f"   Precision: {tensorrt_result['model_info']}")
    else:
        print(f"❌ TensorRT conversion failed: {tensorrt_result.get('error', 'Unknown error')}")
        print("   Note: TensorRT requires NVIDIA GPU and TensorRT installation")
    
    print("\n7. Batch conversion example...")
    print("-" * 40)
    
    # Batch conversion (if you have multiple models)
    batch_result = client.batch_convert(
        model_paths=["models/yolov8n.pt"],
        target_format="onnx",
        imgsz=416,
        batch_size=1,
        dynamic=False
    )
    
    if batch_result.get("success", False):
        print(f"✅ Batch conversion completed!")
        print(f"   Summary: {batch_result['summary']}")
    else:
        print(f"❌ Batch conversion failed: {batch_result.get('error', 'Unknown error')}")
    
    print("\n8. Different image sizes and batch sizes...")
    print("-" * 40)
    
    # Convert with different settings for different use cases
    use_cases = [
        {"name": "Edge deployment", "imgsz": 320, "batch_size": 1, "dynamic": False},
        {"name": "Server inference", "imgsz": 640, "batch_size": 4, "dynamic": False},
        {"name": "Flexible deployment", "imgsz": 640, "batch_size": 1, "dynamic": True},
    ]
    
    for use_case in use_cases:
        print(f"\n   Converting for {use_case['name']}...")
        
        request = {
            "model_path": "models/yolov8n.pt",
            "target_format": "onnx",
            "output_dir": "converted_models",
            **use_case
        }
        del request["name"]  # Remove name from request
        
        result = client.convert_model(request)
        
        if result.get("success", False):
            print(f"   ✅ {use_case['name']}: {Path(result['output_path']).name}")
        else:
            print(f"   ❌ {use_case['name']}: {result.get('error', 'Failed')}")
    
    print(f"\n" + "=" * 60)
    print("API conversion example completed!")
    print("=" * 60)
    
    # Show converted models
    output_dir = Path("converted_models")
    if output_dir.exists():
        print(f"\nConverted models in '{output_dir}':")
        for model_file in sorted(output_dir.glob("*")):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
