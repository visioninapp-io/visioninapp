"""
File storage utility for managing uploaded images and datasets
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from PIL import Image
import uuid


class FileStorage:
    """Handles file upload, validation, and storage operations"""

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, base_upload_dir: str = "uploads"):
        self.base_upload_dir = Path(base_upload_dir)
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.base_upload_dir,
            self.base_upload_dir / "datasets",
            self.base_upload_dir / "models",
            self.base_upload_dir / "temp"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_dataset_directory(self, dataset_id: int) -> Path:
        """Get or create directory for a specific dataset"""
        dataset_dir = self.base_upload_dir / "datasets" / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (dataset_dir / "images").mkdir(exist_ok=True)
        (dataset_dir / "annotations").mkdir(exist_ok=True)

        return dataset_dir

    def validate_image(self, file: UploadFile) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded image file
        Returns: (is_valid, error_message)
        """
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        # Check file size (if available)
        if hasattr(file, 'size') and file.size:
            if file.size > self.MAX_FILE_SIZE:
                return False, f"File too large. Max size: {self.MAX_FILE_SIZE / 1024 / 1024}MB"

        return True, None

    async def save_image(
        self,
        file: UploadFile,
        dataset_id: int,
        validate_content: bool = True
    ) -> dict:
        """
        Save uploaded image file and return metadata

        Args:
            file: Uploaded file
            dataset_id: Dataset ID to associate with
            validate_content: Whether to validate image content using PIL

        Returns:
            dict with file metadata (path, size, dimensions, etc.)
        """
        # Validate file
        is_valid, error_msg = self.validate_image(file)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"

        # Get dataset directory
        dataset_dir = self.get_dataset_directory(dataset_id)
        image_path = dataset_dir / "images" / unique_filename

        # Save file to disk
        try:
            contents = await file.read()

            # Validate image content if requested
            if validate_content:
                try:
                    from io import BytesIO
                    img = Image.open(BytesIO(contents))
                    img.verify()  # Verify it's a valid image

                    # Re-open to get dimensions (verify() closes the file)
                    img = Image.open(BytesIO(contents))
                    width, height = img.size
                    img_format = img.format
                except Exception as e:
                    raise ValueError(f"Invalid image file: {str(e)}")
            else:
                width, height = None, None
                img_format = None

            # Write to disk
            with open(image_path, 'wb') as f:
                f.write(contents)

            # Get file size
            file_size = len(contents)

            return {
                "filename": file.filename,
                "stored_filename": unique_filename,
                "file_path": str(image_path),
                "relative_path": str(image_path.relative_to(self.base_upload_dir)),
                "file_size": file_size,
                "width": width,
                "height": height,
                "format": img_format
            }

        except Exception as e:
            # Clean up on error
            if image_path.exists():
                image_path.unlink()
            raise Exception(f"Failed to save image: {str(e)}")

    async def save_images_batch(
        self,
        files: List[UploadFile],
        dataset_id: int
    ) -> tuple[List[dict], List[dict]]:
        """
        Save multiple images in batch

        Returns:
            (successful_uploads, failed_uploads)
        """
        successful = []
        failed = []

        for file in files:
            try:
                metadata = await self.save_image(file, dataset_id)
                successful.append({
                    "filename": file.filename,
                    "metadata": metadata
                })
            except Exception as e:
                failed.append({
                    "filename": file.filename,
                    "error": str(e)
                })

        return successful, failed

    def delete_dataset_files(self, dataset_id: int):
        """Delete all files associated with a dataset"""
        dataset_dir = self.get_dataset_directory(dataset_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

    def get_image_path(self, dataset_id: int, filename: str) -> Path:
        """Get full path to an image file"""
        return self.get_dataset_directory(dataset_id) / "images" / filename

    def get_model_directory(self, model_id: int) -> Path:
        """Get or create directory for a specific model"""
        model_dir = self.base_upload_dir / "models" / f"model_{model_id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def save_model_file(self, model_id: int, model_data: bytes, filename: str) -> str:
        """Save model file and return path"""
        model_dir = self.get_model_directory(model_id)
        model_path = model_dir / filename

        with open(model_path, 'wb') as f:
            f.write(model_data)

        return str(model_path)


# Singleton instance
file_storage = FileStorage()
