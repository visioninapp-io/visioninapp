"""
Presigned URL service for direct S3 upload/download.
"""
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
from app.utils.content_type_detector import detect_content_type
from typing import Optional, Dict
import uuid
import time


class PresignedURLService:
    """Manage S3 presigned URLs for uploads and downloads."""
    
    def __init__(self):
        """Initialize the S3 client."""
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
            print(f"[PresignedURLService] S3 client initialized for bucket: {settings.AWS_BUCKET_NAME}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
    
    def generate_upload_url(
        self, 
        dataset_id: int, 
        filename: str, 
        content_type: Optional[str] = None,
        expiration: int = 3600,
        asset_kind: Optional[str] = None
    ) -> Dict:
        """
        Generate a presigned URL for uploading a single file.
        
        Args:
            dataset_id: Dataset ID.
            filename: Original filename.
            content_type: MIME type (auto-detected if None).
            expiration: URL expiration in seconds.
            asset_kind: "image" | "video" (auto-detected from content_type if None).
            
        Returns:
            dict with: upload_url, s3_key, unique_filename, original_filename, expires_in, generation_time, kind.
        """
        start_time = time.time()
        
        # Auto-detect content type when not provided
        if content_type is None:
            content_type = detect_content_type(filename)
        
        # Auto-detect asset kind from content type if not provided
        if asset_kind is None:
            if content_type.startswith('video/'):
                asset_kind = 'video'
            else:
                asset_kind = 'image'
        
        # Generate unique filename
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        
        # Build S3 key based on asset kind
        if asset_kind == 'video':
            s3_key = f"datasets/dataset_{dataset_id}/videos/{unique_filename}"
        else:
            s3_key = f"datasets/dataset_{dataset_id}/images/{unique_filename}"
        
        try:
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': settings.AWS_BUCKET_NAME,
                    'Key': s3_key,
                    'ContentType': content_type
                },
                ExpiresIn=expiration
            )
            
            elapsed_time = time.time() - start_time
            print(f"[PresignedURL] Generated upload URL in {elapsed_time:.4f}s for {filename} (kind: {asset_kind})")
            
            return {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "unique_filename": unique_filename,
                "original_filename": filename,
                "kind": asset_kind,
                "content_type": content_type,
                "expires_in": expiration,
                "generation_time": elapsed_time
            }
            
        except ClientError as e:
            raise Exception(f"Failed to generate presigned upload URL: {str(e)}")
    
    def generate_download_url(
        self, 
        s3_key: str, 
        expiration: int = 3600,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Generate a presigned URL for downloading a file.
        
        Args:
            s3_key: S3 object key.
            expiration: URL expiration in seconds.
            filename: Optional download filename.
            
        Returns:
            dict with: download_url, s3_key, expires_in, generation_time.
        """
        start_time = time.time()
        
        # Normalize path for S3 (Windows to Unix)
        s3_key = s3_key.replace("\\", "/")
        
        try:
            params = {
                'Bucket': settings.AWS_BUCKET_NAME,
                'Key': s3_key
            }
            
            # Optionally force download filename
            if filename:
                params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'
            
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expiration
            )
            
            elapsed_time = time.time() - start_time
            print(f"[PresignedURL] Generated download URL in {elapsed_time:.4f}s for {s3_key}")
            
            return {
                "download_url": presigned_url,
                "s3_key": s3_key,
                "expires_in": expiration,
                "generation_time": elapsed_time
            }
            
        except ClientError as e:
            raise Exception(f"Failed to generate presigned download URL: {str(e)}")
    
    def generate_batch_upload_urls(
        self,
        dataset_id: int,
        filenames: list[str],
        content_type: Optional[str] = None,
        expiration: int = 3600
    ) -> list[Dict]:
        """
        Generate presigned upload URLs for multiple files.
        
        Args:
            dataset_id: Dataset ID.
            filenames: List of original filenames.
            content_type: Optional MIME type (auto-detected per file when None).
            expiration: URL expiration in seconds.
            
        Returns:
            list of dicts, one per file.
        """
        start_time = time.time()
        
        results = []
        for filename in filenames:
            try:
                # Detect per-file content type
                file_content_type = content_type if content_type else detect_content_type(filename)
                
                url_data = self.generate_upload_url(
                    dataset_id=dataset_id,
                    filename=filename,
                    content_type=file_content_type,
                    expiration=expiration,
                    asset_kind=None  # auto-detect from content_type
                )
                results.append(url_data)
            except Exception as e:
                results.append({
                    "original_filename": filename,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        print(f"[PresignedURL] Generated {len(results)} upload URLs in {total_time:.4f}s")
        
        return results
    
    def generate_model_upload_url(
        self,
        model_id: int,
        filename: str,
        content_type: str = "application/octet-stream",
        expiration: int = 3600
    ) -> Dict:
        """
        Generate a presigned URL for uploading a model artifact (pt, onnx, etc.).
        
        Args:
            model_id: Model ID.
            filename: Original filename.
            content_type: MIME type (default: octet-stream).
            expiration: URL expiration in seconds.
            
        Returns:
            dict with: upload_url, s3_key, unique_filename, original_filename, expires_in, generation_time.
        """
        start_time = time.time()
        
        # Generate unique filename
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'pt'
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        
        # Build S3 key: models/model_X/artifacts/uuid.ext
        s3_key = f"models/model_{model_id}/artifacts/{unique_filename}"
        
        try:
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': settings.AWS_BUCKET_NAME,
                    'Key': s3_key,
                    'ContentType': content_type
                },
                ExpiresIn=expiration
            )
            
            elapsed_time = time.time() - start_time
            print(f"[PresignedURL] Generated model upload URL in {elapsed_time:.4f}s for {filename}")
            
            return {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "unique_filename": unique_filename,
                "original_filename": filename,
                "expires_in": expiration,
                "generation_time": elapsed_time
            }
            
        except ClientError as e:
            raise Exception(f"Failed to generate presigned model upload URL: {str(e)}")
    
    def check_object_exists(self, s3_key: str) -> bool:
        """
        Check if an S3 object exists.
        
        Args:
            s3_key: S3 object key.
            
        Returns:
            bool indicating existence.
        """
        try:
            s3_key = s3_key.replace("\\", "/")
            self.s3_client.head_object(
                Bucket=settings.AWS_BUCKET_NAME,
                Key=s3_key
            )
            return True
        except ClientError:
            return False


# Singleton instance
presigned_url_service = PresignedURLService()

