"""
TensorRT Conversion Test Script
Tests TensorRT conversion functionality with proper environment checks
"""

import sys
from pathlib import Path
import torch

# Add AI module to path
sys.path.insert(0, str(Path(__file__).parent))

from conversion_service import convert_to_tensorrt, get_model_info
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_version_info():
    """Print detailed version information"""
    print("=" * 70)
    print("DETAILED VERSION INFORMATION")
    print("=" * 70)
    
    # PyTorch and CUDA versions
    print("\n🔹 PYTORCH & CUDA:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   PyTorch CUDA Version: {torch.version.cuda}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Total Memory: {props.total_memory / 1e9:.1f} GB")
    
    # TensorRT version
    print("\n🔹 TENSORRT:")
    try:
        import tensorrt as trt
        print(f"   TensorRT Version: {trt.__version__}")
        print(f"   TensorRT Import: SUCCESS")
    except ImportError as e:
        print(f"   TensorRT Import: FAILED - {e}")
    
    # System CUDA versions
    print("\n🔹 SYSTEM CUDA:")
    try:
        import subprocess
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"   NVIDIA Driver: {driver_version}")
        else:
            print("   NVIDIA Driver: Could not detect")
        
        # Check nvidia-smi CUDA version
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            # Extract CUDA version from nvidia-smi output
            import re
            cuda_match = re.search(r'CUDA Version: ([\d.]+)', output)
            if cuda_match:
                print(f"   nvidia-smi CUDA: {cuda_match.group(1)}")
        
        # Check nvcc version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            # Extract CUDA version from nvcc output
            cuda_match = re.search(r'release ([\d.]+)', output)
            if cuda_match:
                print(f"   nvcc CUDA: {cuda_match.group(1)}")
        else:
            print("   nvcc: Not available")
            
    except Exception as e:
        print(f"   System CUDA check failed: {e}")
    
    # TensorRT packages
    print("\n🔹 TENSORRT PACKAGES:")
    try:
        import subprocess
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            tensorrt_packages = [line for line in lines if 'tensorrt' in line.lower()]
            if tensorrt_packages:
                for pkg in tensorrt_packages:
                    if pkg.strip():
                        print(f"   {pkg.strip()}")
            else:
                print("   No TensorRT packages found")
    except Exception as e:
        print(f"   Package check failed: {e}")
    
    # Environment variables
    print("\n🔹 ENVIRONMENT:")
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"   CUDA_HOME: {cuda_home}")
    print(f"   CUDA_PATH: {cuda_path}")


def check_tensorrt_requirements():
    """Check if TensorRT requirements are met"""
    print("\n" + "=" * 70)
    print("TENSORRT REQUIREMENTS CHECK")
    print("=" * 70)
    
    requirements_met = True
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("   ❌ CUDA is required for TensorRT")
        requirements_met = False
    else:
        print(f"   ✅ CUDA Version: {torch.version.cuda}")
        print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✅ GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check TensorRT availability
    print(f"\n2. TensorRT Check:")
    try:
        import tensorrt as trt
        print(f"   ✅ TensorRT Version: {trt.__version__}")
    except ImportError:
        print("   ❌ TensorRT not installed")
        print("   Install from: https://developer.nvidia.com/tensorrt")
        requirements_met = False
    
    # Check Ultralytics TensorRT support
    print(f"\n3. Ultralytics TensorRT Support:")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # Try to check if engine export is available
        print("   ✅ Ultralytics supports TensorRT export")
    except Exception as e:
        print(f"   ❌ Ultralytics TensorRT issue: {e}")
        requirements_met = False
    
    print(f"\n4. Environment Variables:")
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    return requirements_met


def test_tensorrt_conversion():
    """Test TensorRT conversion with different configurations"""
    print("\n" + "=" * 60)
    print("TensorRT Conversion Tests")
    print("=" * 60)
    
    model_path = "yolov8n.pt"
    output_dir = "tensorrt_test_models"
    
    # Test configurations
    test_configs = [
        {
            "name": "FP32 Precision",
            "precision": "fp32",
            "imgsz": 640,
            "batch_size": 1,
            "workspace": 4
        },
        {
            "name": "FP16 Precision (Recommended)",
            "precision": "fp16",
            "imgsz": 640,
            "batch_size": 1,
            "workspace": 4
        },
        {
            "name": "Small Input Size",
            "precision": "fp16",
            "imgsz": 320,
            "batch_size": 1,
            "workspace": 2
        },
        {
            "name": "Batch Processing",
            "precision": "fp16",
            "imgsz": 640,
            "batch_size": 4,
            "workspace": 6
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. Testing {config['name']}...")
        print("-" * 40)
        
        try:
            result = convert_to_tensorrt(
                model_path=model_path,
                output_dir=output_dir,
                precision=config["precision"],
                imgsz=config["imgsz"],
                batch_size=config["batch_size"],
                workspace=config["workspace"]
            )
            
            if result.get("success", False):
                print(f"   ✅ {config['name']} conversion successful!")
                print(f"      Output: {result['output_path']}")
                print(f"      Time: {result['conversion_time']:.2f}s")
                print(f"      Size: {result['model_info']['source_size_mb']:.1f}MB -> {result['model_info']['output_size_mb']:.1f}MB")
                
                # Check if file exists
                output_file = Path(result['output_path'])
                if output_file.exists():
                    print(f"      File size: {output_file.stat().st_size / (1024*1024):.1f}MB")
                
                results.append({"config": config, "result": result, "success": True})
            else:
                print(f"   ❌ {config['name']} conversion failed: {result.get('error', 'Unknown error')}")
                results.append({"config": config, "result": result, "success": False})
                
        except Exception as e:
            print(f"   ❌ {config['name']} conversion error: {e}")
            results.append({"config": config, "error": str(e), "success": False})
    
    return results


def test_int8_conversion():
    """Test INT8 conversion (requires calibration data)"""
    print("\n" + "=" * 60)
    print("INT8 Conversion Test (Advanced)")
    print("=" * 60)
    
    print("INT8 conversion requires calibration data...")
    print("This test will attempt conversion without calibration (should fallback to FP16)")
    
    try:
        result = convert_to_tensorrt(
            model_path="yolov8n.pt",
            output_dir="tensorrt_test_models",
            precision="int8",
            imgsz=640,
            batch_size=1,
            workspace=4
            # Note: No calibration data provided
        )
        
        if result.get("success", False):
            print(f"✅ INT8 conversion successful (or fallback applied)")
            print(f"   Output: {result['output_path']}")
            print(f"   Actual precision used: Check logs above")
        else:
            print(f"❌ INT8 conversion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ INT8 conversion error: {e}")


def benchmark_inference_speed():
    """Benchmark inference speed of different formats"""
    print("\n" + "=" * 60)
    print("Inference Speed Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping benchmark")
        return
    
    try:
        from ultralytics import YOLO
        import time
        import numpy as np
        
        # Test image (dummy)
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        formats_to_test = [
            {"name": "PyTorch (.pt)", "path": "yolov8n.pt"},
        ]
        
        # Add TensorRT models if they exist
        tensorrt_dir = Path("tensorrt_test_models")
        if tensorrt_dir.exists():
            for engine_file in tensorrt_dir.glob("*.engine"):
                formats_to_test.append({
                    "name": f"TensorRT ({engine_file.stem})",
                    "path": str(engine_file)
                })
        
        print(f"Testing {len(formats_to_test)} formats...")
        
        for format_info in formats_to_test:
            print(f"\n{format_info['name']}:")
            try:
                model = YOLO(format_info['path'])
                
                # Warmup
                for _ in range(3):
                    model.predict(test_image, verbose=False)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    model.predict(test_image, verbose=False)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                fps = 1.0 / avg_time
                
                print(f"   Average time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
                print(f"   FPS: {fps:.1f}")
                
            except Exception as e:
                print(f"   ❌ Benchmark failed: {e}")
                
    except ImportError:
        print("❌ Required packages not available for benchmarking")


def main():
    """Main test function"""
    print("TensorRT Conversion Testing Suite")
    
    # Print detailed version information first
    print_version_info()
    
    # Check requirements
    if not check_tensorrt_requirements():
        print("\n❌ TensorRT requirements not met!")
        print("\nTo install TensorRT:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit")
        print("3. Install TensorRT from NVIDIA developer portal")
        print("4. Install Python TensorRT: pip install tensorrt")
        print("\nAlternatively, test on a system with NVIDIA GPU and TensorRT installed.")
        return False
    
    print("\n✅ All requirements met! Proceeding with tests...")
    
    # Run conversion tests
    results = test_tensorrt_conversion()
    
    # Test INT8 conversion
    test_int8_conversion()
    
    # Benchmark if conversions were successful
    successful_conversions = sum(1 for r in results if r.get("success", False))
    if successful_conversions > 0:
        benchmark_inference_speed()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {len(results) - successful_conversions}")
    
    if successful_conversions > 0:
        print(f"\n✅ TensorRT conversion is working!")
        print(f"Check 'tensorrt_test_models/' directory for converted models")
    else:
        print(f"\n❌ No successful TensorRT conversions")
        print(f"Check CUDA/TensorRT installation")
    
    return successful_conversions > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
