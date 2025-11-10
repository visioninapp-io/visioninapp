"""
Training service for managing PyTorch model training and YOLO training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Callable
import json
from datetime import datetime, timedelta
from app.utils.timezone import get_kst_now_naive


class ObjectDetectionDataset(Dataset):
    """Custom dataset for object detection training"""

    def __init__(self, images_data: list, transform=None):
        """
        Args:
            images_data: List of dicts with 'path', 'annotations' keys
            transform: Optional transform to be applied on images
        """
        self.images_data = images_data
        self.transform = transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        img_data = self.images_data[idx]
        image = Image.open(img_data['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For now, return image and basic label
        # In production, this would include bounding boxes and class labels
        label = img_data.get('label', 0)

        return image, label


class TrainingEngine:
    """Manages training job execution and monitoring"""

    def __init__(self, job_id: int, db_session_factory):
        self.job_id = job_id
        self.db_session_factory = db_session_factory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.is_paused = False
        self.is_cancelled = False
        self.current_epoch = 0

    def _get_db(self):
        """Get database session"""
        return self.db_session_factory()

    def _update_job_status(self, status, **kwargs):
        """Update training job status in database"""
        from app.models.training import TrainingJob, TrainingStatus

        db = self._get_db()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
            if job:
                # Accept both TrainingStatus enum and string
                if isinstance(status, str):
                    job.status = TrainingStatus[status.upper()]
                else:
                    job.status = status
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                db.commit()
        finally:
            db.close()

    def _save_metric(self, epoch: int, metrics: Dict):
        """Save training metrics to database"""
        from app.models.training import TrainingMetric

        db = self._get_db()
        try:
            metric = TrainingMetric(
                training_job_id=self.job_id,
                epoch=epoch,
                step=metrics.get('step', 0),
                train_loss=metrics.get('train_loss', 0.0),
                train_accuracy=metrics.get('train_accuracy', 0.0),
                val_loss=metrics.get('val_loss'),
                val_accuracy=metrics.get('val_accuracy'),
                learning_rate=metrics.get('learning_rate', 0.001)
            )
            db.add(metric)
            db.commit()
        finally:
            db.close()

    def _build_model(self, architecture: str, num_classes: int = 10):
        """Build model based on architecture name"""
        if 'yolo' in architecture.lower():
            # YOLO models use different training path - return None
            # They will be trained via _train_yolo() method
            return None
        elif architecture.lower() == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture.lower() == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture.lower() == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        else:
            # Default to ResNet18
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model.to(self.device)

    def _prepare_data(self, dataset_id: int):
        """Prepare dataset and dataloaders"""
        from app.models.dataset import Dataset as DatasetModel, Image as ImageModel

        db = self._get_db()
        try:
            # Get dataset
            dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")

            # Get all images
            images = db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).all()

            if len(images) == 0:
                raise ValueError(f"No images found in dataset {dataset_id}")

            # Prepare image data
            images_data = []
            base_upload_dir = Path("uploads")
            for img in images:
                img_path = base_upload_dir / img.file_path
                if img_path.exists():
                    images_data.append({
                        'path': str(img_path),
                        'label': 0  # Simplified for now
                    })

            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # Create dataset
            full_dataset = ObjectDetectionDataset(images_data, transform=transform)

            # Split into train/val (80/20)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            return train_dataset, val_dataset

        finally:
            db.close()

    def train(self, config: Dict):
        """
        Execute training job

        Args:
            config: Training configuration dict with keys:
                - dataset_id: Dataset ID
                - architecture: Model architecture name
                - hyperparameters: Dict with batch_size, epochs, learning_rate, etc.
        """
        from app.models.training import TrainingStatus

        try:
            # Update status to running
            self._update_job_status(TrainingStatus.RUNNING, started_at=get_kst_now_naive())

            # Extract config
            dataset_id = config['dataset_id']
            architecture = config['architecture']
            hyperparams = config['hyperparameters']

            # Check if this is a YOLO model - use different training path
            if 'yolo' in architecture.lower():
                print(f"[Training] Detected YOLO architecture: {architecture}")
                print(f"[Training] Using YOLO training pipeline...")
                return self._train_yolo(config)

            # For PyTorch models (classification)
            batch_size = hyperparams.get('batch_size', 32)
            epochs = hyperparams.get('epochs', 10)
            learning_rate = hyperparams.get('learning_rate', 0.001)
            num_classes = hyperparams.get('num_classes', 10)

            # Prepare data
            train_dataset, val_dataset = self._prepare_data(dataset_id)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Set to 0 for Windows compatibility
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

            # Build model
            self.model = self._build_model(architecture, num_classes)

            # Setup optimizer and loss
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(epochs):
                # Check for pause/cancel
                while self.is_paused and not self.is_cancelled:
                    time.sleep(1)

                if self.is_cancelled:
                    self._update_job_status(TrainingStatus.CANCELLED)
                    return

                self.current_epoch = epoch + 1

                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    # Check cancellation during batch
                    if self.is_cancelled:
                        self._update_job_status(TrainingStatus.CANCELLED)
                        return

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = 100.0 * train_correct / train_total
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total

                # Save metrics
                metrics = {
                    'step': batch_idx,
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'learning_rate': learning_rate
                }
                self._save_metric(epoch + 1, metrics)

                # Update job progress
                progress = ((epoch + 1) / epochs) * 100
                estimated_completion = get_kst_now_naive() + timedelta(
                    seconds=(epochs - epoch - 1) * 60  # Rough estimate
                )

                self._update_job_status(
                    TrainingStatus.RUNNING,
                    current_epoch=epoch + 1,
                    current_loss=avg_train_loss,
                    current_accuracy=train_accuracy,
                    current_learning_rate=learning_rate,
                    progress_percentage=progress,
                    estimated_completion=estimated_completion
                )

                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% - "
                      f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%")

            # Training completed
            self._save_model(config)
            self._update_job_status(TrainingStatus.COMPLETED, completed_at=get_kst_now_naive())

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(error_msg)
            self._update_job_status(TrainingStatus.FAILED, training_logs=error_msg)

    def _save_model(self, config: Dict):
        """Save trained model to S3 and create ModelArtifact record"""
        from app.utils.file_storage import file_storage
        from app.models.model import Model as ModelRecord, ModelStatus
        from app.models.model_artifact import ModelArtifact
        from app.models.training import TrainingJob
        import shutil

        db = self._get_db()
        try:
            # Get training job
            job = db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
            if not job or not job.model_id:
                return

            # Save model state
            model_dir = file_storage.get_model_directory(job.model_id)
            
            # Determine if this is a YOLO model or PyTorch model
            architecture = config.get('architecture', '').lower()
            is_yolo = 'yolo' in architecture
            
            if is_yolo:
                # For YOLO models, save as .pt
                model_path = model_dir / "best.pt"
                # If using YOLO, the model should already be saved by ultralytics
                # Just update the path
                print(f"[Training] YOLO model should be saved by ultralytics training")
            else:
                # For PyTorch models, save as .pth
                model_path = model_dir / "model.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': config
                }, model_path)

            # Copy to AI/models/best.pt for easy access by auto-annotation
            if model_path.exists():
                # Get project root by going up from this file's location
                # BE/app/services/training_service.py -> go up to project root
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent.parent
                ai_models_dir = project_root / "AI" / "models"
                ai_models_dir.mkdir(parents=True, exist_ok=True)
                
                # Always save as best.pt in AI/models
                ai_model_path = ai_models_dir / "best.pt"
                
                print(f"[Training] Project root: {project_root}")
                print(f"[Training] Target path: {ai_model_path}")
                
                if is_yolo:
                    # Copy the .pt file
                    shutil.copy2(model_path, ai_model_path)
                    print(f"[Training] ✅ Copied YOLO model to {ai_model_path}")
                else:
                    # For PyTorch models, just note it's not compatible with YOLO
                    print(f"[Training] ⚠️  PyTorch model saved, but won't work with YOLO auto-annotation")
                    print(f"[Training] Use YOLO architecture for auto-annotation compatibility")

            # Update model record
            model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
            if model:
                from app.models.model import ModelStatus
                model.file_path = str(model_path)
                model.file_size = model_path.stat().st_size if model_path.exists() else 0
                model.status = ModelStatus.COMPLETED
                db.commit()

            # Save model state to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': config
                }, tmp_file.name)
                
                tmp_path = Path(tmp_file.name)
                file_size = tmp_path.stat().st_size
                
                # Upload to S3
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                
                # Generate S3 key
                unique_filename = f"{uuid.uuid4()}.pt"
                s3_key = f"models/model_{job.model_id}/artifacts/{unique_filename}"
                
                # Upload file
                with open(tmp_path, 'rb') as f:
                    s3_client.put_object(
                        Bucket=settings.AWS_BUCKET_NAME,
                        Key=s3_key,
                        Body=f,
                        ContentType='application/octet-stream'
                    )
                
                print(f"[Training] Model uploaded to S3: {s3_key}")
                
                # Clean up temp file
                tmp_path.unlink()
            
            # Mark existing primary artifacts as non-primary
            db.query(ModelArtifact).filter(
                ModelArtifact.model_id == job.model_id,
                ModelArtifact.is_primary == True
            ).update({"is_primary": False})
            
            # Create ModelArtifact record
            artifact = ModelArtifact(
                model_id=job.model_id,
                kind='pt',
                version='1.0',
                s3_key=s3_key,
                file_size=file_size,
                is_primary=True,
                created_by=model.created_by
            )
            db.add(artifact)
            
            # Update model record
            model.file_path = s3_key
            model.file_size = file_size
            model.status = ModelStatus.COMPLETED
            
            db.commit()
            print(f"[Training] Model artifact created: {artifact.id}")

        except Exception as e:
            print(f"[Training] Failed to save model to S3: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()

    def pause(self):
        """Pause training"""
        from app.models.training import TrainingStatus
        self.is_paused = True
        self._update_job_status(TrainingStatus.PAUSED)

    def resume(self):
        """Resume training"""
        from app.models.training import TrainingStatus
        self.is_paused = False
        self._update_job_status(TrainingStatus.RUNNING)

    def cancel(self):
        """Cancel training"""
        from app.models.training import TrainingStatus
        self.is_cancelled = True
        self._update_job_status(TrainingStatus.CANCELLED)

    def _train_yolo(self, config: Dict):
        """
        Train YOLO model using Ultralytics
        
        Args:
            config: Training configuration
        """
        from app.models.training import TrainingStatus
        import sys
        
        try:
            print("[YOLO Training] Starting YOLO training pipeline...")
            
            # Import ultralytics
            try:
                from ultralytics import YOLO
                print("[YOLO Training] ✅ Ultralytics imported successfully")
            except ImportError:
                error_msg = "Ultralytics not installed. Run: pip install ultralytics"
                print(f"[YOLO Training] ❌ {error_msg}")
                self._update_job_status(TrainingStatus.FAILED, training_logs=error_msg)
                return
            
            # Check for CUDA availability
            import torch
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                print(f"[YOLO Training] 🚀 CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("[YOLO Training] ⚠️  CUDA not available, using CPU (training will be slower)")
            
            # Prepare YOLO dataset
            data_yaml_path = self._prepare_yolo_dataset(config['dataset_id'])
            if not data_yaml_path:
                error_msg = "Failed to prepare YOLO dataset"
                print(f"[YOLO Training] ❌ {error_msg}")
                self._update_job_status(TrainingStatus.FAILED, training_logs=error_msg)
                return
            
            print(f"[YOLO Training] Dataset prepared: {data_yaml_path}")
            
            # Get hyperparameters
            hyperparams = config['hyperparameters']
            epochs = hyperparams.get('epochs', 100)
            imgsz = hyperparams.get('img_size', 640)
            batch = hyperparams.get('batch_size', 16)
            
            # Get model directory for saving
            from app.models.training import TrainingJob
            from app.utils.file_storage import file_storage
            
            db = self._get_db()
            try:
                job = db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
                if job and job.model_id:
                    model_dir = file_storage.get_model_directory(job.model_id)
                    project_dir = str(model_dir.parent)
                    name = model_dir.name
                else:
                    project_dir = "uploads/models"
                    name = f"yolo_training_{self.job_id}"
            finally:
                db.close()
            
            # Initialize YOLO model
            base_model = "yolov8n.pt"  # Start from pretrained yolov8n
            print(f"[YOLO Training] Loading base model: {base_model}")
            model = YOLO(base_model)
            
            # Setup callback for progress updates
            last_update_epoch = [0]  # Use list to allow modification in nested function
            
            def on_fit_epoch_end(trainer):
                """Callback for epoch end - update database"""
                try:
                    epoch = getattr(trainer, 'epoch', 0)
                    epochs_total = getattr(trainer, 'epochs', epochs)
                    
                    # Debug: Print available attributes (only first epoch)
                    if epoch == 0:
                        print(f"[YOLO Training Debug] Trainer attributes: {[attr for attr in dir(trainer) if not attr.startswith('_')][:20]}")
                    
                    # Get loss from trainer
                    loss = 0.0
                    if hasattr(trainer, 'tloss'):
                        # Total loss (this is the main training loss)
                        tloss_tensor = trainer.tloss
                        if tloss_tensor is not None:
                            # tloss can be a tensor with multiple elements (box, cls, dfl losses)
                            # Sum them to get total loss
                            if hasattr(tloss_tensor, 'sum'):
                                loss = float(tloss_tensor.sum().item())
                            elif hasattr(tloss_tensor, 'item'):
                                loss = float(tloss_tensor.item())
                            else:
                                loss = float(tloss_tensor)
                    elif hasattr(trainer, 'loss'):
                        loss_val = trainer.loss
                        if loss_val is not None:
                            if hasattr(loss_val, 'sum'):
                                loss = float(loss_val.sum().item())
                            elif hasattr(loss_val, 'item'):
                                loss = float(loss_val.item())
                            else:
                                loss = float(loss_val)
                    
                    # Get mAP from trainer metrics (only available after validation)
                    map50 = 0.0
                    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                        metrics = trainer.metrics
                        
                        # Debug: Print metrics structure on first epoch
                        if epoch == 0:
                            print(f"[YOLO Training Debug] Metrics type: {type(metrics)}")
                            print(f"[YOLO Training Debug] Metrics attrs: {[a for a in dir(metrics) if not a.startswith('_')][:15]}")
                        
                        # Try to get map50 from DetMetrics object
                        if hasattr(metrics, 'box') and metrics.box is not None:
                            box_metrics = metrics.box
                            if hasattr(box_metrics, 'map50'):
                                map50 = float(box_metrics.map50)
                            elif hasattr(box_metrics, 'all_ap'):
                                # Fallback to all_ap if map50 not available
                                map50 = float(box_metrics.all_ap)
                        
                        # Try results_dict as fallback
                        elif hasattr(metrics, 'results_dict') and metrics.results_dict:
                            results = metrics.results_dict
                            map50 = float(results.get('metrics/mAP50(B)', 
                                              results.get('mAP50',
                                              results.get('map50', 0.0))))
                    
                    # Update database with progress
                    progress = ((epoch + 1) / epochs_total) * 100
                    
                    self._update_job_status(
                        TrainingStatus.RUNNING,
                        current_epoch=epoch + 1,
                        current_loss=loss,
                        current_accuracy=map50 * 100,  # Use mAP as "accuracy"
                        progress_percentage=progress,
                        estimated_completion=get_kst_now_naive() + timedelta(
                            seconds=(epochs_total - epoch - 1) * 60
                        )
                    )
                    
                    # Save metric to database
                    self._save_metric(epoch + 1, {
                        'step': 0,
                        'train_loss': loss,
                        'train_accuracy': map50 * 100,
                        'val_loss': None,
                        'val_accuracy': None,
                        'learning_rate': 0.01
                    })
                    
                    last_update_epoch[0] = epoch + 1
                    print(f"[YOLO Training] Epoch {epoch + 1}/{epochs_total} - Loss: {loss:.4f}, mAP50: {map50:.3f}, Progress: {progress:.1f}%")
                    
                except Exception as e:
                    print(f"[YOLO Training] ⚠️  Error in progress callback: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add callback to model (correct way for Ultralytics)
            model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
            print(f"[YOLO Training] ✅ Progress callback registered")
            
            # Train
            print(f"[YOLO Training] Starting training...")
            print(f"[YOLO Training]   Device: {device}")
            print(f"[YOLO Training]   Epochs: {epochs}")
            print(f"[YOLO Training]   Image size: {imgsz}")
            print(f"[YOLO Training]   Batch size: {batch}")
            print(f"[YOLO Training]   Data: {data_yaml_path}")
            
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,  # Use CUDA if available
                project=project_dir,
                name=name,
                exist_ok=True,
                verbose=True
            )
            
            print("[YOLO Training] ✅ Training completed!")
            
            # Copy best model to AI/models/best.pt
            weights_dir = Path(project_dir) / name / "weights"
            best_pt = weights_dir / "best.pt"
            
            if best_pt.exists():
                # Get project root
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent.parent
                ai_models_dir = project_root / "AI" / "models"
                ai_models_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy to AI/models/best.pt
                ai_best_path = ai_models_dir / "best.pt"
                import shutil
                shutil.copy2(best_pt, ai_best_path)
                
                print(f"[YOLO Training] ✅ Copied best model to {ai_best_path}")
                
                # Update model record
                from app.models.model import Model as ModelRecord, ModelStatus
                db = self._get_db()
                try:
                    job = db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
                    if job and job.model_id:
                        model_record = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
                        if model_record:
                            model_record.file_path = str(best_pt)
                            model_record.file_size = best_pt.stat().st_size
                            model_record.status = ModelStatus.COMPLETED
                            db.commit()
                            print(f"[YOLO Training] ✅ Updated model record")
                finally:
                    db.close()
            else:
                print(f"[YOLO Training] ⚠️ Best model not found at {best_pt}")
            
            # Update job status
            self._update_job_status(TrainingStatus.COMPLETED, completed_at=get_kst_now_naive())
            print("[YOLO Training] ✅ All done!")
            
        except Exception as e:
            error_msg = f"YOLO training failed: {str(e)}"
            print(f"[YOLO Training] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            self._update_job_status(TrainingStatus.FAILED, training_logs=error_msg)

    def _prepare_yolo_dataset(self, dataset_id: int) -> Optional[str]:
        """
        Prepare dataset in YOLO format and create data.yaml
        
        Args:
            dataset_id: Dataset ID from database
            
        Returns:
            Path to data.yaml file, or None if failed
        """
        from app.models.dataset import Dataset as DatasetModel, Image as ImageModel, Annotation
        from pathlib import Path
        import yaml
        
        db = self._get_db()
        try:
            # Get dataset
            dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
            if not dataset:
                print(f"[YOLO Dataset] ❌ Dataset {dataset_id} not found")
                return None
            
            print(f"[YOLO Dataset] Preparing dataset: {dataset.name}")
            
            # Get all images with annotations
            images = db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).all()
            if not images:
                print(f"[YOLO Dataset] ❌ No images found")
                return None
            
            print(f"[YOLO Dataset] Found {len(images)} images")
            
            # Create YOLO dataset directory structure
            yolo_dataset_dir = Path("uploads") / "yolo_datasets" / f"dataset_{dataset_id}"
            yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
            
            for split in ['train', 'val']:
                (yolo_dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
                (yolo_dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
            # Split images (80% train, 20% val)
            import random
            random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            print(f"[YOLO Dataset] Split: {len(train_images)} train, {len(val_images)} val")
            
            # Process images and create label files
            base_upload_dir = Path("uploads")
            
            for split, img_list in [('train', train_images), ('val', val_images)]:
                for img in img_list:
                    # Get image path
                    src_img_path = base_upload_dir / img.file_path
                    if not src_img_path.exists():
                        print(f"[YOLO Dataset] ⚠️ Image not found: {src_img_path}")
                        continue
                    
                    # Copy image to YOLO structure
                    dst_img_path = yolo_dataset_dir / 'images' / split / img.filename
                    import shutil
                    shutil.copy2(src_img_path, dst_img_path)
                    
                    # Get annotations
                    annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
                    
                    # Create label file (YOLO format)
                    label_filename = Path(img.filename).stem + '.txt'
                    label_path = yolo_dataset_dir / 'labels' / split / label_filename
                    
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                            f.write(f"{ann.class_id} {ann.x_center} {ann.y_center} {ann.width} {ann.height}\n")
            
            # Get class names from latest dataset version's label ontology
            from app.models.label_class import LabelClass
            from app.models.dataset import DatasetVersion
            
            latest_version = db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == dataset.id
            ).order_by(DatasetVersion.created_at.desc()).first()
            
            class_names = []
            if latest_version and latest_version.ontology_version:
                label_classes = db.query(LabelClass).filter(
                    LabelClass.ontology_version_id == latest_version.ontology_version_id
                ).all()
                class_names = [lc.display_name for lc in label_classes]
            
            num_classes = len(class_names) if class_names else 1
            
            data_yaml = {
                'path': str(yolo_dataset_dir.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': num_classes,
                'names': class_names if class_names else ['object']
            }
            
            data_yaml_path = yolo_dataset_dir / 'data.yaml'
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            
            print(f"[YOLO Dataset] ✅ Dataset prepared at {yolo_dataset_dir}")
            print(f"[YOLO Dataset] ✅ data.yaml created with {num_classes} classes: {class_names}")
            
            return str(data_yaml_path)
            
        except Exception as e:
            print(f"[YOLO Dataset] ❌ Error preparing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            db.close()


class TrainingManager:
    """Manages multiple training jobs"""

    def __init__(self):
        self.active_jobs: Dict[int, TrainingEngine] = {}

    def start_training(self, job_id: int, config: Dict, db_session_factory):
        """Start training job in background thread"""
        if job_id in self.active_jobs:
            raise ValueError(f"Training job {job_id} is already running")

        engine = TrainingEngine(job_id, db_session_factory)
        self.active_jobs[job_id] = engine

        # Start training in background thread
        thread = threading.Thread(
            target=engine.train,
            args=(config,),
            daemon=True
        )
        thread.start()

        return engine

    def pause_training(self, job_id: int):
        """Pause a training job"""
        if job_id not in self.active_jobs:
            raise ValueError(f"Training job {job_id} is not running")

        self.active_jobs[job_id].pause()

    def resume_training(self, job_id: int):
        """Resume a paused training job"""
        if job_id not in self.active_jobs:
            raise ValueError(f"Training job {job_id} is not running")

        self.active_jobs[job_id].resume()

    def cancel_training(self, job_id: int):
        """Cancel a training job"""
        if job_id not in self.active_jobs:
            raise ValueError(f"Training job {job_id} is not running")

        self.active_jobs[job_id].cancel()
        del self.active_jobs[job_id]

    def get_active_jobs(self):
        """Get list of active job IDs"""
        return list(self.active_jobs.keys())


# Global training manager instance
training_manager = TrainingManager()