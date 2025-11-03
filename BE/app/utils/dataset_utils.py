"""
Dataset utility functions for YOLO training
"""

from pathlib import Path
from typing import Optional
import yaml
import shutil
import random
import logging

logger = logging.getLogger(__name__)


def prepare_yolo_dataset(dataset_id: int, db_session_factory) -> Optional[str]:
    """
    Prepare YOLO dataset from database images and annotations
    
    Args:
        dataset_id: Dataset ID to prepare
        db_session_factory: Database session factory (SessionLocal)
        
    Returns:
        Path to data.yaml file, or None if failed
    """
    from app.models.dataset import Dataset as DatasetModel, Image as ImageModel, Annotation
    
    db = db_session_factory()
    try:
        # Get dataset
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            logger.error(f"[YOLO Dataset] Dataset {dataset_id} not found")
            return None
        
        logger.info(f"[YOLO Dataset] Preparing dataset: {dataset.name}")
        
        # Get images
        images = db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).all()
        if not images:
            logger.error(f"[YOLO Dataset] No images found")
            return None
        
        logger.info(f"[YOLO Dataset] Found {len(images)} images")
        
        # Create YOLO dataset directory
        yolo_dataset_dir = Path("uploads") / "yolo_datasets" / f"dataset_{dataset_id}"
        yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train/val directories
        for split in ['train', 'val']:
            (yolo_dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Split dataset (80/20)
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        logger.info(f"[YOLO Dataset] Split: {len(train_images)} train, {len(val_images)} val")
        
        base_upload_dir = Path("uploads")
        
        # Process images and create labels
        for split, img_list in [('train', train_images), ('val', val_images)]:
            for img in img_list:
                src_img_path = base_upload_dir / img.file_path
                if not src_img_path.exists():
                    logger.warning(f"[YOLO Dataset] Image not found: {src_img_path}")
                    continue
                
                # Copy image
                dst_img_path = yolo_dataset_dir / 'images' / split / img.filename
                shutil.copy2(src_img_path, dst_img_path)
                
                # Get annotations
                annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
                
                # Create label file
                label_filename = Path(img.filename).stem + '.txt'
                label_path = yolo_dataset_dir / 'labels' / split / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        # YOLO format: class_id x_center y_center width height (normalized)
                        f.write(f"{ann.class_id} {ann.x_center} {ann.y_center} {ann.width} {ann.height}\n")
        
        # Get class names
        class_names = dataset.class_names if dataset.class_names else []
        num_classes = len(class_names) if class_names else 1
        
        # Create data.yaml
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
        
        # Return absolute path so AI service can find it
        absolute_path = data_yaml_path.absolute()
        
        logger.info(f"[YOLO Dataset] ✅ Dataset prepared at {yolo_dataset_dir}")
        logger.info(f"[YOLO Dataset] ✅ data.yaml created with {num_classes} classes: {class_names}")
        logger.info(f"[YOLO Dataset] ✅ Absolute path: {absolute_path}")
        
        return str(absolute_path)
        
    except Exception as e:
        logger.error(f"[YOLO Dataset] Error preparing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

