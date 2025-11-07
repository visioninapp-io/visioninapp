# AI Module - Model Training & Inference

This module provides unified interfaces for model training and inference, with built-in support for YOLO models via Ultralytics.

## Features

- 🏋️ **Message-driven training service** with real-time progress tracking
- 🔄 **Model conversion service** for ONNX and TensorRT formats
- 🔍 **Inference service** for auto-annotation and object detection
- 🐰 **RabbitMQ integration** for async job processing (training/conversion/inference)
- 🔌 **Pure consumer/producer architecture** - no HTTP server overhead
- 📊 **Real-time updates** sent automatically during processing
- 📦 **Automatic model downloading** (YOLOv8n fallback)
- ⚡ **Multi-service support** with unified message handling
- 🎯 **Production-ready** with proper error handling and logging

## Quick Start

### Installation

```bash
cd AI
pip install -r requirements.txt
```

### Configuration

The AI service now runs as a pure message-driven service. Configure RabbitMQ:

```bash
# Set environment variables or create .env file
export ENABLE_RABBITMQ=true
export RABBITMQ_HOST=localhost
export RABBITMQ_USER=guest
export RABBITMQ_PASSWORD=guest
```

### Starting the Service

```bash
# Using startup script (recommended)
./start.sh

# Or directly
python ai_service.py
```

The service will:
- ✅ Connect to RabbitMQ
- 🎯 Listen for training requests on `train_request_q`
- 🔄 Listen for conversion requests on `convert_request_q`
- 🔍 Listen for inference requests on `inference_request_q`
- 📤 Send real-time updates to respective update/result queues
- 📋 Send final results back to backend

### Testing the Service

Use the comprehensive test suite to verify the message-driven service:

```bash
# Interactive test runner (recommended)
python test_runner.py

# Or run tests directly from tests/ directory
cd tests
python run_tests.py

# Individual test files
python test_individual_functions.py  # Unit tests (no RabbitMQ)
python test_service_methods.py      # Service method tests
python test_multi_service.py        # Full integration tests
python test_message_service.py      # Training-only tests
```

See `tests/README.md` for detailed testing documentation.

The multi-service test will:
- 🔍 **Test inference**: Auto-annotation on a sample image
- 🔄 **Test conversion**: Convert YOLOv8n to ONNX format
- 🏋️ **Test training**: Train a model for 3 epochs
- 👂 **Listen for real-time updates** from all services
- 📊 **Display progress and final results**
- ✅ **Validate complete message flows**

### Manual Testing

You can also send requests manually to any service:

#### Training Request:

```python
import pika
import json

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Send training request
message = {
    "job_id": "my_job_123",
    "dataset": {"name": "coco128.yaml"},
    "hyperparams": {
        "model": "yolov8n",
        "epochs": 10,
        "batch": 16,
        "imgsz": 640
    },
    "output": {
        "prefix": "runs/train",
        "model_name": "my_model.pt"
    }
}

channel.basic_publish(
    exchange='',
    routing_key='train_request_q',
    body=json.dumps(message)
)
```

#### Conversion Request:

```python
# Convert model to ONNX
conversion_message = {
    "job_id": "convert_123",
    "model_path": "yolov8n.pt",
    "target_format": "onnx",
    "options": {
        "imgsz": 640,
        "batch_size": 1,
        "dynamic": False,
        "simplify": True
    }
}

channel.basic_publish(
    exchange='',
    routing_key='convert_request_q',
    body=json.dumps(conversion_message)
)
```

#### Inference Request:

```python
# Run inference on image
inference_message = {
    "job_id": "inference_123",
    "image_path": "path/to/image.jpg",
    "model_path": "yolov8n.pt",  # Optional, uses default if not specified
    "options": {
        "conf": 0.25,
        "iou": 0.45
    }
}

channel.basic_publish(
    exchange='',
    routing_key='inference_request_q',
    body=json.dumps(inference_message)
)
```

## Architecture

The AI service is a pure message-driven service using RabbitMQ for all communication:

```
┌─────────────┐    RabbitMQ Queues   ┌─────────────┐
│   Backend   │ ──────────────────►  │ AI Service  │
│             │  train_request_q     │             │
│             │  convert_request_q   │ Training    │
│             │  inference_request_q │ Conversion  │
│             │                      │ Inference   │
│             │ ◄──────────────────  │             │
└─────────────┘  *_result_q queues   └─────────────┘
```

### File Structure:

```
AI/
├── model_trainer/          # Core training utilities
│   ├── integrations/
│   │   └── yolo.py        # YOLO adapter with progress tracking
│   ├── factory.py         # Model factory
│   ├── trainer.py         # Training orchestration
│   └── payload.py         # Payload-based API
├── training_service.py    # High-level training API
├── conversion_service.py # Model conversion service
├── inference_service.py  # Inference and auto-annotation service
├── ai_service.py         # Message-driven AI service (main)
├── multi_service_consumer.py # Multi-service RabbitMQ consumer
├── rabbitmq_producer.py  # RabbitMQ producer
├── config.py             # Configuration management
├── examples/              # Usage examples
│   └── be_consumer_example.py   # Backend consumer example
├── tests/               # Comprehensive test suite
│   ├── test_individual_functions.py # Unit tests
│   ├── test_service_methods.py      # Service method tests
│   ├── test_multi_service.py        # Integration tests
│   ├── test_message_service.py      # Training tests
│   ├── run_tests.py                 # Interactive test runner
│   └── README.md                    # Testing documentation
├── test_runner.py          # Main test runner
├── start.sh              # Service startup script
└── requirements.txt      # Multi-service dependencies
```

## Backend Integration

### Auto-Annotation Service (Already Integrated)

The BE auto-annotation service (`BE/app/services/auto_annotation_service.py`) now automatically:
1. Tries to load custom model from `AI/models/best.pt`
2. Falls back to `yolov8n.pt` if no custom model exists
3. Auto-downloads yolov8n on first use

```python
# In BE/app/services/auto_annotation_service.py
service = get_auto_annotation_service()
service.load_model()  # Will use yolov8n if no custom model

# Run inference
annotations = service.predict_image(image_path, conf_threshold=0.25)
```

### Training Service Integration (Optional Enhancement)

To integrate AI training service into BE:

```python
import sys
sys.path.insert(0, 'path/to/AI')

from AI.training_service import YOLOTrainingService

class EnhancedTrainingEngine:
    def train_yolo(self, config):
        service = YOLOTrainingService('yolov8n.pt')
        
        def progress_callback(status):
            # Update database with progress
            if status['event'] == 'epoch_end':
                self._save_metric(status['epoch'], status['metrics'])
        
        results = service.train(
            data_yaml=config['data_yaml'],
            epochs=config['epochs'],
            progress_callback=progress_callback
        )
        
        return results
```

## Available Models

### Pre-trained YOLO Models (Auto-download)

- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

### Custom Models

Train your own model and save to `AI/models/best.pt` for automatic use by the backend.

## Dataset Format (YOLO)

### Directory Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

### Label Format (YOLO)

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```
All values are normalized to 0-1.

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

### data.yaml Configuration

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

nc: 2  # number of classes
names: ['class1', 'class2']
```

## Model Conversion

The AI module now supports converting YOLO models to optimized formats for deployment:

### Supported Formats

- **ONNX**: Cross-platform format for edge devices and ONNX Runtime
- **TensorRT**: NVIDIA GPU-optimized format for high-performance inference

### Conversion Options

#### ONNX Conversion
```python
from AI.conversion_service import convert_to_onnx

result = convert_to_onnx(
    model_path="models/best.pt",
    output_dir="converted_models",
    imgsz=640,           # Input image size
    batch_size=1,        # Batch size
    dynamic=False,       # Enable dynamic shapes
    simplify=True,       # Simplify ONNX graph
    opset=17            # ONNX opset version
)
```

#### TensorRT Conversion
```python
from AI.conversion_service import convert_to_tensorrt

result = convert_to_tensorrt(
    model_path="models/best.pt",
    output_dir="converted_models",
    imgsz=640,           # Input image size
    batch_size=1,        # Batch size
    precision="fp16",    # fp32, fp16, or int8
    workspace=4,         # GPU workspace (GB)
    int8_calibration_data="path/to/calibration.yaml"  # For INT8
)
```

### API Endpoints

The FastAPI service provides REST endpoints for model conversion:

```bash
# Get supported formats
GET /api/v1/convert/formats

# Get model information
GET /api/v1/models/{model_name}/info

# Convert a model
POST /api/v1/convert
{
  "model_path": "models/yolov8n.pt",
  "target_format": "onnx",
  "imgsz": 640,
  "batch_size": 1,
  "dynamic": false
}

# Batch conversion
POST /api/v1/convert/batch
{
  "model_paths": ["model1.pt", "model2.pt"],
  "target_format": "onnx",
  "conversion_options": {...}
}
```

### Use Cases

| Format | Use Case | Benefits | Requirements |
|--------|----------|----------|--------------|
| ONNX | Edge deployment, Cross-platform | Universal compatibility, Smaller size | ONNX Runtime |
| TensorRT | NVIDIA GPU servers | Maximum performance, Low latency | NVIDIA GPU, TensorRT |

### Performance Comparison

| Format | Inference Speed | Model Size | Compatibility |
|--------|----------------|------------|---------------|
| PyTorch (.pt) | Baseline | Largest | PyTorch only |
| ONNX | 1.2-2x faster | 10-30% smaller | Universal |
| TensorRT FP16 | 2-4x faster | 50% smaller | NVIDIA GPU |
| TensorRT INT8 | 3-6x faster | 75% smaller | NVIDIA GPU + calibration |

## Examples

Run the example scripts to see the services in action:

```bash
# Inference examples
python AI/examples/inference_example.py

# Training examples
python AI/examples/training_example.py

# Basic YOLO training
python AI/examples/train_yolo.py

# Model conversion examples
python AI/examples/conversion_example.py

# API-based conversion examples
python AI/examples/api_conversion_example.py
```

## API Reference

### Inference Service

```python
class YOLOInferenceService:
    def __init__(self, model_spec: str = "yolov8n.pt")
    def load_model() -> bool
    def predict(source, conf=0.25, iou=0.45, **kwargs) -> List[Dict]
    def get_model_info() -> Dict
```

### Training Service

```python
class YOLOTrainingService:
    def __init__(self, model_spec: str = "yolov8n.pt")
    def train(
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict
    def get_best_model_path() -> Optional[Path]
```

### Conversion Service

```python
class ModelConversionService:
    def __init__(self)
    def convert_model(
        model_path: Union[str, Path],
        target_format: str,
        output_dir: Optional[Union[str, Path]] = None,
        **conversion_options
    ) -> Dict[str, Any]
    def get_conversion_info(model_path: Union[str, Path]) -> Dict[str, Any]
    def batch_convert(
        model_paths: List[Union[str, Path]],
        target_format: str,
        output_dir: Optional[Union[str, Path]] = None,
        **conversion_options
    ) -> List[Dict[str, Any]]

# Convenience functions
def convert_to_onnx(model_path, output_dir=None, **options) -> Dict
def convert_to_tensorrt(model_path, output_dir=None, **options) -> Dict
def get_model_info(model_path) -> Dict
```

### Progress Callback Format

```python
def progress_callback(status: Dict):
    # status['event'] can be:
    # - 'epoch_end': After each epoch completes
    # - 'tick': Every second during training
    # - 'train_end': When training completes
    
    # Available fields:
    # - epoch: Current epoch number
    # - total_epochs: Total epochs
    # - metrics: Dictionary of training metrics
    # - save_dir: Directory where model is saved
```

## Troubleshooting

### Model Download Issues

If automatic download fails, manually download models from:
https://github.com/ultralytics/assets/releases

Place them in your home directory under `.cache/torch/hub/checkpoints/` or specify full path.

### Import Errors

Make sure to add the AI directory to Python path:
```python
import sys
sys.path.insert(0, 'path/to/AI')
```

### GPU not detected

Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Advanced Usage

### Custom Training Parameters

```python
service = YOLOTrainingService("yolov8n.pt")
results = service.train(
    data_yaml='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,           # Initial learning rate
    momentum=0.937,     # SGD momentum
    weight_decay=0.0005,# Weight decay
    warmup_epochs=3,    # Warmup epochs
    patience=50,        # Early stopping patience
    save_period=10,     # Save checkpoint every N epochs
    workers=8,          # Dataloader workers
    device=0,           # GPU device (0, 1, 2, ... or 'cpu')
)
```

### Multiple GPU Training

```python
results = service.train(
    data_yaml='data.yaml',
    epochs=100,
    device=[0, 1, 2, 3],  # Use 4 GPUs
)
```

### Resume Training

```python
service = YOLOTrainingService("path/to/last.pt")
results = service.train(
    data_yaml='data.yaml',
    resume=True,
    epochs=200  # Continue to 200 total epochs
)
```

## Integration with BE/FE

### Flow Diagram

```
FE (JavaScript)
    ↓ API Call
BE (FastAPI)
    ↓ Load Service
AI (Python Module)
    ↓ Inference/Training
Ultralytics YOLO
```

### Current Integration Points

1. **Auto-Annotation**: BE → AI inference service → Ultralytics
2. **Training** (optional): BE → AI training service → Ultralytics
3. **Model Management**: BE stores models, AI uses them

## Contributing

When adding new model frameworks:
1. Create adapter in `model_trainer/integrations/`
2. Add factory method in `inference_service.py` or `training_service.py`
3. Update examples and tests

## License

See main project LICENSE file.
