# VisionAI Platform

A comprehensive computer vision platform inspired by Roboflow, providing end-to-end MLOps workflow for object detection models. Built with FastAPI backend and Vanilla JavaScript frontend.

## 🚀 Features

### 📊 Dataset Management
- **Upload & Organize**: Batch upload images with drag-and-drop support
- **Auto-Annotation**: AI-powered automatic labeling using YOLO models
- **Label Management**: Define classes with custom colors and metadata
- **Smart Annotation Tools**: Bounding box and polygon annotation support

### 🔄 Data Augmentation & Versioning
- **Dataset Versions**: Create multiple versions with different preprocessing/augmentation settings
- **Preprocessing Options**:
  - Auto-orient, resize (fit/stretch/pad)
  - Grayscale conversion
  - Contrast and exposure normalization
  - Static crop and tiling
- **Augmentation Techniques**:
  - Geometric: Flip, rotate, crop, shear
  - Color: Hue, saturation, brightness, exposure
  - Effects: Blur, noise, cutout
- **Data Splitting**: Customizable train/valid/test split ratios

### 📦 Export Formats
Export datasets in multiple industry-standard formats:
- **YOLOv8/v5**: YOLO format with data.yaml
- **COCO JSON**: Microsoft COCO format
- **Pascal VOC**: XML-based format
- **CSV**: Simple comma-separated format
- **TFRecord**: TensorFlow format
- **CreateML**: Apple ML format

### 🤖 Model Training
- **Multiple Architectures**: YOLOv5, YOLOv8, YOLOv11, Faster R-CNN, SSD, EfficientDet
- **Hyperparameter Tuning**: Customizable learning rate, batch size, epochs, optimizer
- **Real-time Monitoring**: Live training metrics and progress tracking
- **Transfer Learning**: Use pre-trained weights or train from scratch

### 📈 Model Evaluation
- **Comprehensive Metrics**: Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95
- **Per-Class Performance**: Detailed metrics for each object class
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Model Comparison**: Compare multiple models side-by-side

### 🌐 Model Deployment
- **Local Inference API**: Deploy models as REST API endpoints
- **Hosted Inference**: Cloud-based inference with auto-scaling
- **Performance Monitoring**: Track request counts, response times, uptime
- **API Key Management**: Secure access with auto-generated API keys

### 📊 Monitoring & Analytics
- **Real-time Metrics**: Monitor deployed model performance
- **Alerts & Notifications**: Configure alerts for performance degradation
- **Edge Case Detection**: Automatically identify challenging samples
- **Feedback Loop**: Continuous learning from production data

## 🏗️ Architecture

```
final_pjt/
├── BE/                     # Backend (FastAPI)
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core configs (auth, database, cache)
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   ├── main.py            # FastAPI application
│   └── requirements.txt
├── FE/                     # Frontend (Vanilla JS + Bootstrap)
│   ├── js/
│   │   ├── pages/         # Page components
│   │   ├── services/      # API & Firebase services
│   │   └── app.js         # Router
│   ├── css/
│   └── index.html
└── AI/                     # Pre-trained models (.pt files)
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **Database**: SQLite (development), PostgreSQL (production)
- **Auth**: Firebase Admin SDK
- **Cache**: Redis (optional)
- **ML**: PyTorch, Ultralytics (YOLO), OpenCV

### Frontend
- **Core**: Vanilla JavaScript (ES6+)
- **UI**: Bootstrap 5
- **Charts**: Chart.js
- **Auth**: Firebase JS SDK
- **Icons**: Bootstrap Icons

## 📋 Prerequisites

- Python 3.8+
- Node.js (for development server)
- Git

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd final_pjt
```

### 2. Backend Setup

```bash
# Navigate to backend
cd BE

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir uploads augmented exports

# Run the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

### 3. Frontend Setup

```bash
# Navigate to frontend
cd FE

# Option 1: Use Live Server (VS Code extension)
# Right-click index.html -> Open with Live Server

# Option 2: Use Python HTTP server
python -m http.server 8080

# Option 3: Use Node.js http-server
npx http-server -p 8080
```

The frontend will be available at `http://localhost:8080`

### 4. Firebase Configuration (Optional)

If you want to use Firebase authentication:

1. Create a Firebase project at https://console.firebase.google.com/
2. Download service account credentials
3. Save as `BE/firebase-credentials.json`
4. Update `FE/js/config/firebase-config.js` with your Firebase config

For development, the app works without Firebase using a mock user.

## 📁 Project Structure

### Backend API Endpoints

```
/api/v1/
├── datasets/              # Dataset CRUD operations
│   ├── POST /             # Create dataset
│   ├── GET /              # List datasets
│   ├── GET /{id}          # Get dataset
│   ├── POST /{id}/upload  # Upload images
│   ├── POST /{id}/auto-annotate  # Auto-annotate
│   └── /{id}/versions     # Dataset versions
├── export/                # Export datasets
│   ├── POST /             # Create export job
│   ├── GET /              # List exports
│   └── GET /{id}/download # Download export
├── training/              # Model training
│   ├── POST /             # Start training
│   ├── GET /              # List training jobs
│   └── GET /{id}/metrics  # Get training metrics
├── models/                # Model management
│   ├── GET /              # List models
│   ├── POST /{id}/convert # Convert model format
│   └── POST /{id}/predict # Run inference
├── evaluation/            # Model evaluation
│   ├── POST /             # Create evaluation
│   └── GET /{id}          # Get evaluation results
├── deployment/            # Model deployment
│   ├── POST /             # Create deployment
│   ├── POST /{id}/inference  # Run inference
│   └── GET /{id}/logs     # Get inference logs
└── monitoring/            # Performance monitoring
    ├── GET /alerts        # Get alerts
    └── GET /metrics       # Get performance metrics
```

### Frontend Pages

```
/#/                        # Home dashboard
/#/datasets                # Dataset management
/#/generate                # Data augmentation & versions
/#/export                  # Export datasets
/#/training                # Model training
/#/conversion              # Model format conversion
/#/evaluation              # Model evaluation
/#/deployment              # Model deployment
/#/monitoring              # Performance monitoring
```

## 🔧 Configuration

### Backend Configuration (`BE/app/core/config.py`)

```python
# API Settings
PROJECT_NAME = "VisionAI Platform"
API_V1_STR = "/api/v1"

# CORS
ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# Database
DATABASE_URL = "sqlite:///./app.db"

# File Storage
UPLOAD_DIR = "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Cache (Optional)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ENABLE_CACHE = False
```

### Frontend Configuration (`FE/js/services/api.js`)

```javascript
const API_BASE_URL = 'http://localhost:8000/api/v1';
```

## 📖 Usage Examples

### 1. Create and Upload Dataset

```python
# Using Python requests
import requests

# Create dataset
response = requests.post('http://localhost:8000/api/v1/datasets/', json={
    "name": "Factory Defects",
    "description": "Manufacturing defect detection",
    "total_classes": 2,
    "class_names": ["good", "defect"]
})
dataset = response.json()

# Upload images
files = [('files', open('image1.jpg', 'rb')), ('files', open('image2.jpg', 'rb'))]
requests.post(f'http://localhost:8000/api/v1/datasets/{dataset["id"]}/upload', files=files)
```

### 2. Create Augmented Version

```python
version_data = {
    "name": "v1-augmented",
    "train_split": 0.7,
    "valid_split": 0.2,
    "test_split": 0.1,
    "preprocessing_config": {
        "resize": {"width": 640, "height": 640, "mode": "fit"},
        "auto_orient": True,
        "grayscale": False
    },
    "augmentation_config": {
        "output_count": 3,
        "flip_horizontal": 0.5,
        "rotate": {"min": -15, "max": 15},
        "brightness": {"min": -20, "max": 20}
    }
}

response = requests.post(
    f'http://localhost:8000/api/v1/datasets/{dataset_id}/versions',
    json=version_data
)
```

### 3. Export Dataset

```python
export_data = {
    "dataset_id": 1,
    "export_format": "yolov8",
    "include_images": True
}

response = requests.post('http://localhost:8000/api/v1/export/', json=export_data)
export_job = response.json()

# Download when complete
requests.get(f'http://localhost:8000/api/v1/export/{export_job["id"]}/download')
```

### 4. Train Model

```python
training_data = {
    "name": "DefectNet-v1",
    "dataset_id": 1,
    "architecture": "yolov8m",
    "hyperparameters": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "img_size": 640,
        "optimizer": "adam"
    }
}

response = requests.post('http://localhost:8000/api/v1/training/', json=training_data)
```

## 🎯 Roadmap

- [x] Dataset management and upload
- [x] Auto-annotation with YOLO
- [x] Data augmentation and versioning
- [x] Multi-format export (YOLO, COCO, VOC, etc.)
- [x] Model training with multiple architectures
- [x] Comprehensive evaluation metrics
- [x] Model deployment API
- [x] Performance monitoring
- [ ] Real-time collaboration
- [ ] Active learning pipelines
- [ ] Mobile app support
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Multi-GPU training
- [ ] Video annotation support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Inspired by [Roboflow](https://roboflow.com/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [Ultralytics](https://ultralytics.com/)
- UI components from [Bootstrap](https://getbootstrap.com/)

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

---

**Made with ❤️ for Computer Vision Engineers**
