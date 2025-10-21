"""
Data Augmentation Service for Dataset Versions

Handles preprocessing and augmentation of images according to configuration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil
from sqlalchemy.orm import Session


class AugmentationService:
    """Service for applying preprocessing and augmentations to datasets"""

    def __init__(self):
        self.base_dir = Path("uploads")
        self.augmented_dir = Path("augmented")
        self.augmented_dir.mkdir(exist_ok=True)

    def generate_version(
        self,
        version_id: int,
        dataset_id: int,
        preprocessing_config: Dict[str, Any],
        augmentation_config: Dict[str, Any],
        train_split: float,
        valid_split: float,
        test_split: float,
        db: Session
    ) -> Dict[str, int]:
        """
        Generate a dataset version with preprocessing and augmentation

        Returns dict with image counts per split
        """
        from app.models.dataset import Dataset, Image as ImageModel

        # Get dataset and images
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        images = db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).all()
        if not images:
            raise ValueError(f"No images in dataset {dataset_id}")

        # Create version directory
        version_dir = self.augmented_dir / f"dataset_{dataset_id}" / f"version_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Split images
        random.shuffle(images)
        total = len(images)
        train_count = int(total * train_split)
        valid_count = int(total * valid_split)

        train_images = images[:train_count]
        valid_images = images[train_count:train_count + valid_count]
        test_images = images[train_count + valid_count:]

        # Process each split
        splits = {
            "train": train_images,
            "valid": valid_images,
            "test": test_images
        }

        total_processed = 0
        result = {
            "total_images": 0,
            "train_images": 0,
            "valid_images": 0,
            "test_images": 0
        }

        for split_name, split_images in splits.items():
            split_dir = version_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for img_model in split_images:
                # Apply preprocessing
                img_path = self.base_dir / img_model.file_path
                if not img_path.exists():
                    continue

                processed_img = self._apply_preprocessing(str(img_path), preprocessing_config)
                if processed_img is None:
                    continue

                # Save original processed image
                output_path = split_dir / img_model.filename
                cv2.imwrite(str(output_path), processed_img)
                result[f"{split_name}_images"] += 1

                # Apply augmentations (only for train split)
                if split_name == "train" and augmentation_config:
                    augmented_images = self._apply_augmentations(
                        processed_img,
                        augmentation_config
                    )

                    for idx, aug_img in enumerate(augmented_images):
                        aug_filename = f"aug_{idx}_{img_model.filename}"
                        aug_path = split_dir / aug_filename
                        cv2.imwrite(str(aug_path), aug_img)
                        result[f"{split_name}_images"] += 1

        result["total_images"] = sum([
            result["train_images"],
            result["valid_images"],
            result["test_images"]
        ])

        return result

    def _apply_preprocessing(self, img_path: str, config: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing steps to an image"""
        if not config:
            img = cv2.imread(img_path)
            return img

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Auto-orient (using EXIF data - simplified version)
        if config.get("auto_orient", True):
            # For simplicity, we skip EXIF rotation here
            pass

        # Resize
        if "resize" in config and config["resize"]:
            resize_cfg = config["resize"]
            target_width = resize_cfg.get("width", 640)
            target_height = resize_cfg.get("height", 640)
            mode = resize_cfg.get("mode", "fit")  # fit, stretch, pad

            if mode == "stretch":
                img = cv2.resize(img, (target_width, target_height))
            elif mode == "fit":
                img = self._resize_keep_aspect(img, target_width, target_height)
            elif mode == "pad":
                img = self._resize_with_padding(img, target_width, target_height)

        # Grayscale
        if config.get("grayscale", False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Normalize contrast
        if config.get("normalize_contrast", False):
            img = self._normalize_contrast(img)

        # Normalize exposure
        if config.get("normalize_exposure", False):
            img = self._normalize_exposure(img)

        return img

    def _apply_augmentations(
        self,
        img: np.ndarray,
        config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Apply augmentation transformations to an image"""
        output_count = config.get("output_count", 1)
        augmented_images = []

        for _ in range(output_count):
            aug_img = img.copy()

            # Geometric transformations
            if config.get("flip_horizontal") and random.random() < config["flip_horizontal"]:
                aug_img = cv2.flip(aug_img, 1)

            if config.get("flip_vertical") and random.random() < config["flip_vertical"]:
                aug_img = cv2.flip(aug_img, 0)

            if "rotate" in config and config["rotate"]:
                angle = random.uniform(
                    config["rotate"].get("min", -15),
                    config["rotate"].get("max", 15)
                )
                aug_img = self._rotate_image(aug_img, angle)

            if "crop" in config and config["crop"]:
                crop_pct = random.uniform(0, config["crop"].get("max", 20))
                aug_img = self._random_crop(aug_img, crop_pct)

            # Color adjustments (convert to PIL for easier manipulation)
            pil_img = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))

            if "hue" in config and config["hue"]:
                # Hue adjustment is complex, skip for now
                pass

            if "saturation" in config and config["saturation"]:
                factor = 1.0 + random.uniform(
                    config["saturation"].get("min", -25) / 100,
                    config["saturation"].get("max", 25) / 100
                )
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(factor)

            if "brightness" in config and config["brightness"]:
                factor = 1.0 + random.uniform(
                    config["brightness"].get("min", -25) / 100,
                    config["brightness"].get("max", 25) / 100
                )
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(factor)

            if "blur" in config and config["blur"]:
                blur_amount = random.uniform(0, config["blur"].get("max", 2.5))
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_amount))

            # Convert back to OpenCV format
            aug_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Add noise
            if "noise" in config and config["noise"]:
                noise_pct = random.uniform(0, config["noise"].get("max", 5))
                aug_img = self._add_noise(aug_img, noise_pct)

            augmented_images.append(aug_img)

        return augmented_images

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize image keeping aspect ratio (fit inside target)"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    @staticmethod
    def _resize_with_padding(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize with padding to exact dimensions"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    @staticmethod
    def _normalize_contrast(img: np.ndarray) -> np.ndarray:
        """Normalize image contrast using CLAHE"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _normalize_exposure(img: np.ndarray) -> np.ndarray:
        """Normalize image exposure"""
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))

    @staticmethod
    def _random_crop(img: np.ndarray, crop_percent: float) -> np.ndarray:
        """Randomly crop image by percentage"""
        h, w = img.shape[:2]
        crop_h = int(h * crop_percent / 100)
        crop_w = int(w * crop_percent / 100)

        y = random.randint(0, crop_h) if crop_h > 0 else 0
        x = random.randint(0, crop_w) if crop_w > 0 else 0

        return img[y:h - crop_h + y, x:w - crop_w + x]

    @staticmethod
    def _add_noise(img: np.ndarray, noise_percent: float) -> np.ndarray:
        """Add random noise to image"""
        noise = np.random.randint(
            -int(255 * noise_percent / 100),
            int(255 * noise_percent / 100),
            img.shape,
            dtype=np.int16
        )
        noisy = img.astype(np.int16) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
