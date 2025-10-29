"""
File storage utility for managing uploaded images and datasets
S3-only mode: All files are uploaded directly to S3
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from PIL import Image
import uuid
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


class FileStorage:
    """Handles file upload, validation, and storage operations (S3-only)"""

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, base_upload_dir: str = "uploads"):
        """Initialize S3 client for file storage"""
        self.base_upload_dir = Path(base_upload_dir)
        
        # Initialize S3 client (required)
        if not settings.USE_S3_STORAGE:
            raise RuntimeError("S3 storage is required. Set USE_S3_STORAGE=true in .env file")
        
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise RuntimeError("AWS credentials are required")
        
        if not settings.AWS_BUCKET_NAME:
            raise RuntimeError("AWS bucket name is required")
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            print(f"[FileStorage] S3 client initialized for bucket: {settings.AWS_BUCKET_NAME}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
        
        # Still need local directory for temp operations
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
        Upload image file to S3 and return metadata

        Args:
            file: Uploaded file
            dataset_id: Dataset ID to associate with
            validate_content: Whether to validate image content using PIL

        Returns:
            dict with file metadata (s3_key, size, dimensions, etc.)
        """
        # Validate file
        is_valid, error_msg = self.validate_image(file)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"

        # S3 key: datasets/dataset_X/images/filename.jpg
        s3_key = f"datasets/dataset_{dataset_id}/images/{unique_filename}"

        # Read file contents
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

            # Determine content type
            content_type = 'image/jpeg'
            if file_ext in ['.png']:
                content_type = 'image/png'
            elif file_ext in ['.gif']:
                content_type = 'image/gif'
            elif file_ext in ['.bmp']:
                content_type = 'image/bmp'
            elif file_ext in ['.tiff', '.tif']:
                content_type = 'image/tiff'

            # Upload to S3
            self.s3_client.put_object(
                Bucket=settings.AWS_BUCKET_NAME,
                Key=s3_key,
                Body=contents,
                ContentType=content_type
            )

            print(f"[S3] Uploaded: {s3_key}")

            # Get file size
            file_size = len(contents)

            # Relative path for database (same format as before for compatibility)
            relative_path = f"datasets/dataset_{dataset_id}/images/{unique_filename}"

            return {
                "filename": file.filename,
                "stored_filename": unique_filename,
                "file_path": f"datasets/dataset_{dataset_id}/images/{unique_filename}",  # Full S3 path
                "relative_path": relative_path,
                "file_size": file_size,
                "width": width,
                "height": height,
                "format": img_format,
                "s3_key": s3_key,
                "s3_bucket": settings.AWS_BUCKET_NAME
            }

        except ClientError as e:
            raise Exception(f"Failed to upload to S3: {str(e)}")
        except Exception as e:
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

    def download_from_s3(self, s3_key: str) -> Optional[bytes]:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 key/path of the file (e.g., 'datasets/dataset_1/images/image.jpg')
            
        Returns:
            File content as bytes, or None if download fails
        """
        try:
            # Windows 경로를 Unix 경로로 변환
            s3_key = s3_key.replace("\\", "/")
            
            response = self.s3_client.get_object(
                Bucket=settings.AWS_BUCKET_NAME,
                Key=s3_key
            )
            
            file_data = response['Body'].read()
            print(f"[S3] Downloaded file: {s3_key} ({len(file_data)} bytes)")
            return file_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                print(f"[S3] File not found: {s3_key}")
            else:
                print(f"[S3] Error downloading {s3_key}: {e}")
            return None
        except Exception as e:
            print(f"[S3] Unexpected error downloading {s3_key}: {e}")
            return None

    def delete_dataset_files(self, dataset_id: int):
        """Delete all files associated with a dataset from S3"""
        try:
            # S3에서 dataset의 모든 파일 삭제
            prefix = f"datasets/dataset_{dataset_id}/"
            
            # 해당 prefix로 시작하는 모든 객체 조회
            response = self.s3_client.list_objects_v2(
                Bucket=settings.AWS_BUCKET_NAME,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                # 삭제할 객체 목록 생성
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                if objects_to_delete:
                    # 일괄 삭제
                    self.s3_client.delete_objects(
                        Bucket=settings.AWS_BUCKET_NAME,
                        Delete={'Objects': objects_to_delete}
                    )
                    print(f"[S3] Deleted {len(objects_to_delete)} files for dataset {dataset_id}")
            else:
                print(f"[S3] No files found for dataset {dataset_id}")
                
        except ClientError as e:
            print(f"[S3] Error deleting dataset files: {e}")
            raise Exception(f"Failed to delete dataset files from S3: {str(e)}")

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
