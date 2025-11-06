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
import re


def sanitize_dataset_name_for_s3(name: str) -> str:
    """
    데이터셋 이름을 S3 key에 사용 가능하도록 정리
    
    Args:
        name: 원본 데이터셋 이름
        
    Returns:
        S3 key에 사용 가능한 정리된 이름
    """
    # 공백을 언더스코어로 변경
    name = name.replace(' ', '_')
    # 특수문자를 언더스코어로 변경 (알파벳, 숫자, 하이픈, 점, 언더스코어만 허용)
    name = re.sub(r'[^\w\-.]', '_', name)
    # 연속된 언더스코어를 하나로
    name = re.sub(r'_+', '_', name)
    # 양쪽 언더스코어 제거
    name = name.strip('_')
    # 비어있으면 기본값
    if not name:
        name = 'dataset'
    return name


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
        dataset_name: str,
        filename: str,
        asset_number: int,
        content_type: Optional[str] = None,
        expiration: int = 3600,
        asset_kind: Optional[str] = None
    ) -> Dict:
        """
        Generate a presigned URL for uploading a single file.
        
        Args:
            dataset_id: Dataset ID.
            dataset_name: Dataset name (used in S3 path).
            filename: Original filename.
            asset_number: Sequential number for the asset (e.g., 1, 2, 3...).
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
        
        # Get file extension
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'jpg'
        
        # Sanitize dataset name for S3 path
        safe_dataset_name = sanitize_dataset_name_for_s3(dataset_name)
        
        # Generate filename: {dataset_name}_{number}.{ext}
        unique_filename = f"{safe_dataset_name}_{asset_number}.{file_ext}"
        
        # Build S3 key based on asset kind (format: datasets/{name}/images or videos/)
        if asset_kind == 'video':
            s3_key = f"datasets/{safe_dataset_name}/videos/{unique_filename}"
        else:
            s3_key = f"datasets/{safe_dataset_name}/images/{unique_filename}"
        
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
                "original_filename": unique_filename,  # S3 파일명과 동일하게 반환
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
        dataset_name: str,
        filenames: list[str],
        start_number: int = 1,
        content_type: Optional[str] = None,
        expiration: int = 3600
    ) -> list[Dict]:
        """
        Generate presigned upload URLs for multiple files.
        
        Args:
            dataset_id: Dataset ID.
            dataset_name: Dataset name (used in S3 path).
            filenames: List of original filenames.
            start_number: Starting number for sequential filenames (default: 1).
            content_type: Optional MIME type (auto-detected per file when None).
            expiration: URL expiration in seconds.
            
        Returns:
            list of dicts, one per file.
        """
        start_time = time.time()
        
        results = []
        current_number = start_number
        
        for filename in filenames:
            try:
                # Detect per-file content type
                file_content_type = content_type if content_type else detect_content_type(filename)
                
                url_data = self.generate_upload_url(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    filename=filename,
                    asset_number=current_number,
                    content_type=file_content_type,
                    expiration=expiration,
                    asset_kind=None  # auto-detect from content_type
                )
                results.append(url_data)
                current_number += 1
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

    def generate_label_upload_url(
        self,
        dataset_name: str,
        image_filename: str,
        expiration: int = 3600
    ) -> Dict:
        """
        Label 파일 업로드용 Presigned URL 생성

        Label 파일은 datasets/{dataset_name}/labels/{base_filename}.txt 경로에 저장
        이미지 파일명과 매핑됨 (예: image1.jpg -> image1.txt)

        Args:
            dataset_name: 데이터셋 이름 (S3 경로에 사용)
            image_filename: 이미지 파일명 (예: "image1.jpg")
            expiration: URL 만료 시간(초)

        Returns:
            Dict with upload_url, s3_key, filename 등
        """
        start_time = time.time()

        # 데이터셋 이름 정리
        sanitized_dataset_name = sanitize_dataset_name_for_s3(dataset_name)

        # 이미지 파일명에서 확장자 제거하고 .txt로 변경
        base_filename = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
        label_filename = f"{base_filename}.txt"

        # S3 키 생성: datasets/{dataset_name}/labels/{base_filename}.txt
        s3_key = f"datasets/{sanitized_dataset_name}/labels/{label_filename}"

        try:
            # Presigned URL 생성 (ContentType 제외하여 CORS 이슈 방지)
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': settings.AWS_BUCKET_NAME,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )

            elapsed_time = time.time() - start_time
            print(f"[PresignedURL] Label upload URL 생성 완료 ({elapsed_time:.4f}s): {label_filename}")

            return {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "filename": label_filename,
                "image_filename": image_filename,
                "expires_in": expiration,
                "generation_time": elapsed_time
            }

        except ClientError as e:
            raise Exception(f"Label upload URL 생성 실패: {str(e)}")

    def delete_label_file(
        self,
        dataset_name: str,
        image_filename: str
    ) -> bool:
        """
        S3에서 Label 파일 삭제

        Args:
            dataset_name: 데이터셋 이름 (S3 경로에 사용)
            image_filename: 이미지 파일명 (예: "image1.jpg")

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 데이터셋 이름 정리
            sanitized_dataset_name = sanitize_dataset_name_for_s3(dataset_name)

            # 이미지 파일명에서 확장자 제거하고 .txt로 변경
            base_filename = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
            label_filename = f"{base_filename}.txt"

            # S3 키 생성
            s3_key = f"datasets/{sanitized_dataset_name}/labels/{label_filename}"

            # S3에서 삭제
            self.s3_client.delete_object(
                Bucket=settings.AWS_BUCKET_NAME,
                Key=s3_key
            )

            print(f"[PresignedURL] Label 파일 삭제 완료: {s3_key}")
            return True

        except ClientError as e:
            print(f"[PresignedURL] Label 파일 삭제 실패: {str(e)}")
            return False


# Singleton instance
presigned_url_service = PresignedURLService()

