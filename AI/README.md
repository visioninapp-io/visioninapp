<<<<<<< HEAD
# AI Module - Model Training & Inference

This module provides unified interfaces for model training and inference, with built-in support for YOLO models via Ultralytics.

## Features

- 🚀 **Easy-to-use inference service** with automatic model downloading
- 🏋️ **Training service** with progress callbacks and metrics tracking
- 🔌 **Seamless integration** with Backend and Frontend
- 📦 **Automatic fallback** to YOLOv8n when no custom model is available
- 📊 **Real-time progress tracking** during training and inference

## Quick Start

### Installation

```bash
cd AI
pip install -r requirements.txt
```

### 1. Inference (Auto-Annotation)

```python
from AI.inference_service import YOLOInferenceService

# Create service (will auto-download yolov8n if needed)
service = YOLOInferenceService("yolov8n.pt")
service.load_model()

# Run inference
results = service.predict("path/to/image.jpg", conf=0.25)

# Results format:
# [{
#     'class_id': 0,
#     'class_name': 'person',
#     'confidence': 0.89,
#     'bbox': {'x_center': 0.5, 'y_center': 0.5, 'width': 0.3, 'height': 0.4},
#     'bbox_xyxy': [100, 200, 300, 400]
# }, ...]
```

### 2. Training

```python
from AI.training_service import YOLOTrainingService

# Create training service
service = YOLOTrainingService("yolov8n.pt")

# Train with progress callback
def progress_callback(status):
    if status['event'] == 'epoch_end':
        print(f"Epoch {status['epoch']}/{status['total_epochs']}")
        print(f"Metrics: {status['metrics']}")

results = service.train(
    data_yaml='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    progress_callback=progress_callback
)

if results['success']:
    best_model = service.get_best_model_path()
    print(f"Model saved to: {best_model}")
```

## Architecture

```
AI/
├── model_trainer/          # Core training utilities
│   ├── integrations/
│   │   └── yolo.py        # YOLO adapter with progress tracking
│   ├── factory.py         # Model factory
│   ├── trainer.py         # Training orchestration
│   └── payload.py         # Payload-based API
├── inference_service.py   # High-level inference API
├── training_service.py    # High-level training API
├── examples/              # Usage examples
│   ├── train_yolo.py      # Basic training example
│   ├── inference_example.py
│   └── training_example.py
└── requirements.txt
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

## Examples

Run the example scripts to see the services in action:

```bash
# Inference examples
python AI/examples/inference_example.py

# Training examples
python AI/examples/training_example.py

# Basic YOLO training
python AI/examples/train_yolo.py
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
=======
YOLO Trainer (AI.model_trainer)
===============================

Minimal utilities to select a YOLO model (alias or .pt checkpoint), apply hyperparameters, and train via a simple payload API or inside FastAPI.

Quick start
-----------

```bash
pip install -r requirements.txt
python -m AI.examples.train_yolo
```

One-call payload training
-------------------------

```python
from AI.model_trainer import train_from_payload

# Detection (Ultralytics YOLO)
payload = {
    "model": "yolov8n",  # or "/abs/path/model.pt"
    "hyperparameters": {"epochs": 50, "imgsz": 640, "batch": 16},
    "fit_params": {"data": "/abs/path/data.yaml"}
}
trained, score = train_from_payload(payload)
```

Progress and metrics
--------------------

You can receive epoch updates and final metrics (e.g., precision/recall/mAP50-95) via a callback:

```python
from AI.model_trainer import build_model, train_model

events = []

def on_progress(ev):
    # ev = {"event": "epoch_end"|"train_end", "epoch": int, "total_epochs": int, "metrics": {...}, "save_dir": str}
    events.append(ev)
    print(ev)

yolo = build_model("yolov8n", {"epochs": 10, "imgsz": 640})
train_model(yolo, None, None, fit_params={
    "data": "/abs/path/data.yaml",
    "progress_callback": on_progress,
    "tick_interval": 1.0  # send a 'tick' event every second (default 1.0)
})

# After training
print("Final metrics:", getattr(yolo, "final_metrics", None))
print("Runs saved under:", getattr(yolo, "save_dir", None))
```

Retrieving progress (backend snapshots + polling)
------------------------------------------------

Recommended pattern:
- Use `progress_callback` to update a per-job snapshot in your backend (in-memory or Redis).
- Frontend polls `GET /jobs/{id}` every 2–5s to retrieve the latest snapshot.

Example (FastAPI skeleton):
```python
from typing import Any, Dict
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from AI.model_trainer import train_from_payload

app = FastAPI()
jobs: Dict[str, Dict[str, Any]] = {}

class TrainPayload(BaseModel):
    model: str
    hyperparameters: Dict[str, Any] | None = None
    fit_params: Dict[str, Any] | None = None

def on_progress_factory(job_id: str):
    def on_progress(ev: Dict[str, Any]):
        # Store only the latest snapshot
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "running", "progress": ev, "updated_at": __import__("time").time()}
    return on_progress

def _run(job_id: str, payload: Dict[str, Any]):
    jobs[job_id] = {"status": "running"}
    try:
        # inject server-side callback
        p = dict(payload)
        fp = dict(p.get("fit_params") or {})
        fp["progress_callback"] = on_progress_factory(job_id)
        p["fit_params"] = fp
        _, score = train_from_payload(p)
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "completed", "score": score}
    except Exception as e:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "failed", "error": str(e)}

@app.post("/train")
def train(payload: TrainPayload, bg: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued"}
    bg.add_task(_run, job_id, payload.dict(exclude_none=True))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    return jobs[job_id]
```

Optional push (SSE)
-------------------
If you prefer pushing updates, expose an SSE endpoint and `yield` each `on_progress` event to connected clients. Polling is simpler and adequate if you don’t need real-time streaming.

Model selection
---------------

- YOLO alias: "yolov8n", "yolov8s", etc. (auto-resolves to .pt)
- YOLO checkpoint: absolute path to `.pt`

Hyperparameters vs fit_params (YOLO)
------------------------------------

- hyperparameters: build-time settings (epochs, imgsz, batch, etc.).
- fit_params: training-time args passed to Ultralytics (REQUIRED: `data` YAML path).

```python
from AI.model_trainer import build_model, train_model

yolo = build_model("yolov8n", {"epochs": 50, "imgsz": 640, "batch": 16})
train_model(yolo, None, None, fit_params={"data": "/abs/path/data.yaml"})
```

FastAPI usage (payload-based)
-----------------------------

```python
from typing import Any, Dict
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from AI.model_trainer import train_from_payload

app = FastAPI()
jobs: Dict[str, Dict[str, Any]] = {}

class TrainPayload(BaseModel):
    model: str  # 'yolov8n' or '/path/model.pt'
    hyperparameters: Dict[str, Any] | None = None
    fit_params: Dict[str, Any] | None = None

def _run(job_id: str, payload: Dict[str, Any]):
    jobs[job_id] = {"status": "running"}
    try:
        _, score = train_from_payload(payload)
        jobs[job_id] = {"status": "completed", "score": score}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/train")
def train(payload: TrainPayload, bg: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued"}
    bg.add_task(_run, job_id, payload.dict(exclude_none=True))
    return {"job_id": job_id}
```

Notes
-----

- `X_train`/`y_train` are ignored for YOLO; pass dataset YAML via `fit_params.data`.
- To use a custom model (e.g., YOLOv8 variant or your `model.pt`), set `model` accordingly.
- Trained artifacts are managed by Ultralytics in `runs/` unless overridden.


>>>>>>> feature/llm-pipeline
