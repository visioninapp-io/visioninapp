# AI Models Directory

This directory stores trained YOLO models that can be used for auto-annotation.

## Model File Naming

- **`best.pt`** - The best trained model (used by auto-annotation by default)
- **`yolov8n.pt`** - Auto-downloaded YOLOv8 nano model (if no custom model exists)
- Other `.pt` files - Various trained models

## Important Notes

### ✅ Correct Format: `.pt` (YOLO format)
Auto-annotation requires YOLO format models (`.pt` extension).

### ❌ Wrong Format: `.pth` (PyTorch format)
PyTorch models (`.pth`) are **not compatible** with YOLO auto-annotation.

## Training YOLO Models

### Option 1: Using AI Module (Recommended)

```python
from AI.training_service import YOLOTrainingService

# Create service
service = YOLOTrainingService("yolov8n.pt")

# Train
results = service.train(
    data_yaml='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='AI/models',
    name='my_model'
)

# Get best model
best_model = service.get_best_model_path()
# Copy to AI/models/best.pt
import shutil
shutil.copy(best_model, 'AI/models/best.pt')
```

### Option 2: Direct Ultralytics

```python
from ultralytics import YOLO

# Load base model
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='AI/models',
    name='training_run'
)

# Best model will be at: AI/models/training_run/weights/best.pt
# Copy to: AI/models/best.pt
```

### Option 3: Using Backend Training (Current Limitation)

The current backend training service uses PyTorch directly and produces `.pth` files, which are **not compatible** with YOLO auto-annotation.

**To use backend training with YOLO:**
1. Select "yolov8" as architecture
2. After training completes, the system will attempt to copy the model
3. However, the current implementation doesn't fully integrate YOLO training yet

**Workaround until YOLO integration:**
- Train using Option 1 or 2 above
- Or use the pre-downloaded yolov8n.pt for auto-annotation

## Dataset Format for YOLO Training

### Directory Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        └── img3.txt
```

### data.yaml Configuration

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

nc: 2  # number of classes
names: ['class1', 'class2']
```

### Label Format

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```

All values (except class_id) are normalized to 0-1.

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

## Checking Your Model

To verify your model is in the correct format:

```python
from ultralytics import YOLO

# Try to load
model = YOLO("AI/models/best.pt")

# Check classes
print(f"Classes: {model.names}")

# Try prediction
results = model.predict("test_image.jpg")
```

## Auto-Annotation Behavior

1. **Custom model exists (`AI/models/best.pt`):**
   - Uses your trained model ✅
   
2. **Custom model missing:**
   - Falls back to yolov8n.pt (auto-downloads) ✅
   - Shows warning in logs
   
3. **Wrong format (.pth):**
   - Detects incompatibility ⚠️
   - Falls back to yolov8n.pt
   - Shows warning message

## Converting PyTorch to YOLO

Unfortunately, there's no automatic conversion from PyTorch `.pth` to YOLO `.pt`.

**If you have a .pth model:**
- Retrain using YOLO architecture
- Or use a different auto-annotation approach

## Troubleshooting

### "Model not found" Error
- Check if `AI/models/best.pt` exists
- If not, system will auto-download yolov8n.pt

### ".pth acceptable suffix is {'.pt'}" Error
- Your model is in PyTorch format, not YOLO format
- Retrain with YOLO or use yolov8n.pt

### "Classes don't match" Warning
- Your trained model has different classes than your dataset
- This is expected for general models like yolov8n
- Train a custom model on your specific data

## Model Management

### Adding a New Model
```bash
# Copy trained model
cp /path/to/trained/model.pt AI/models/best.pt
```

### Backup Old Models
```bash
# Rename before replacing
mv AI/models/best.pt AI/models/best_backup_$(date +%Y%m%d).pt
```

### List Models
```bash
ls -lh AI/models/*.pt
```

## File Size Reference

| Model | Size | Use Case |
|-------|------|----------|
| yolov8n.pt | ~6 MB | Default, fast inference |
| yolov8s.pt | ~22 MB | Better accuracy |
| yolov8m.pt | ~50 MB | High accuracy |
| Custom | Varies | Your specific dataset |

## See Also

- `AI/README.md` - AI module documentation
- `AI/examples/training_example.py` - Training examples
- `INTEGRATION_GUIDE.md` - Full system integration guide

