"""
Example script demonstrating how to use the AI module for training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_service import YOLOTrainingService, train_yolo
import logging

logging.basicConfig(level=logging.INFO)


def example_basic_training():
    """Basic training example"""
    print("=" * 60)
    print("Example 1: Basic YOLO Training")
    print("=" * 60)
    
    print("""
To train a YOLO model:

1. Prepare your dataset in YOLO format:
   dataset/
     ├── images/
     │   ├── train/
     │   └── val/
     └── labels/
         ├── train/
         └── val/

2. Create a data.yaml file:
   ```yaml
   path: /path/to/dataset
   train: images/train
   val: images/val
   
   nc: 2  # number of classes
   names: ['class1', 'class2']
   ```

3. Run training:
   ```python
   from AI.training_service import YOLOTrainingService
   
   service = YOLOTrainingService('yolov8n.pt')
   results = service.train(
       data_yaml='path/to/data.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       name='my_model'
   )
   
   if results['success']:
       print(f"Model saved to: {results['save_dir']}")
       best_model = service.get_best_model_path()
       print(f"Best model: {best_model}")
   ```
""")


def example_with_progress_callback():
    """Example with progress tracking"""
    print("\n" + "=" * 60)
    print("Example 2: Training with Progress Callback")
    print("=" * 60)
    
    print("""
To track training progress in real-time:

```python
def progress_callback(status):
    event = status.get('event')
    
    if event == 'epoch_end':
        epoch = status.get('epoch')
        total = status.get('total_epochs')
        metrics = status.get('metrics', {})
        print(f"Epoch {epoch}/{total}: {metrics}")
    
    elif event == 'tick':
        # Called every second during training
        epoch = status.get('epoch')
        print(f"Training... Epoch {epoch}")
    
    elif event == 'train_end':
        metrics = status.get('metrics', {})
        print(f"Training completed! Final metrics: {metrics}")

service = YOLOTrainingService('yolov8n.pt')
results = service.train(
    data_yaml='data.yaml',
    epochs=100,
    progress_callback=progress_callback
)
```
""")


def example_integration_with_be():
    """Example showing how BE should integrate for training"""
    print("\n" + "=" * 60)
    print("Example 3: Backend Integration Pattern")
    print("=" * 60)
    
    print("""
Backend Integration for Training:

```python
# In BE training service
import sys
sys.path.insert(0, 'path/to/AI')

from AI.training_service import YOLOTrainingService

class EnhancedTrainingEngine:
    def __init__(self, job_id, db_session_factory):
        self.job_id = job_id
        self.db_session_factory = db_session_factory
        self.yolo_service = None
    
    def train_yolo_model(self, config):
        '''Train YOLO model using AI module'''
        
        # Create YOLO training service
        model_spec = config.get('base_model', 'yolov8n.pt')
        self.yolo_service = YOLOTrainingService(model_spec)
        
        # Progress callback to update database
        def progress_callback(status):
            if status.get('event') == 'epoch_end':
                epoch = status.get('epoch')
                metrics = status.get('metrics', {})
                self._save_metric(epoch, metrics)
                self._update_job_status('RUNNING', 
                                       current_epoch=epoch)
        
        # Prepare data.yaml from dataset
        data_yaml = self._prepare_dataset_yaml(
            config['dataset_id']
        )
        
        # Train
        results = self.yolo_service.train(
            data_yaml=data_yaml,
            epochs=config.get('epochs', 100),
            imgsz=config.get('imgsz', 640),
            batch=config.get('batch', 16),
            name=f"job_{self.job_id}",
            project='uploads/models',
            progress_callback=progress_callback
        )
        
        if results['success']:
            # Save model path to database
            best_model = self.yolo_service.get_best_model_path()
            self._save_model_path(best_model)
            return True
        
        return False
```
""")


def example_quick_training():
    """Example using convenience function"""
    print("\n" + "=" * 60)
    print("Example 4: Quick Training (Convenience Function)")
    print("=" * 60)
    
    print("""
For quick one-off training:

```python
from AI.training_service import train_yolo

results = train_yolo(
    data_yaml='data.yaml',
    model_spec='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    name='quick_train'
)

if results['success']:
    print(f"Training complete! Model at: {results['save_dir']}")
else:
    print(f"Training failed: {results['error']}")
```
""")


def main():
    """Run all examples"""
    print("AI Module Training Examples")
    print("=" * 60)
    
    example_basic_training()
    example_with_progress_callback()
    example_integration_with_be()
    example_quick_training()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nFor actual training, you need:")
    print("  1. A dataset in YOLO format")
    print("  2. A data.yaml configuration file")
    print("  3. Sufficient disk space and GPU (recommended)")
    print("=" * 60)


if __name__ == "__main__":
    main()

