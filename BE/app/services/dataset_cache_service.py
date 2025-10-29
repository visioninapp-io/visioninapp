"""
Dataset caching service for image data with Redis and S3
S3-only mode: All images are stored in and retrieved from S3
Optimized with parallel downloads using ThreadPoolExecutor
"""
import base64
import io
import time
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import settings
from app.services.redis_client import redis_client


class DatasetCacheService:
    """Service for caching dataset images with base64 encoding (S3-only)"""
    
    def __init__(self):
        """Initialize S3 client (required)"""
        if not settings.USE_S3_STORAGE:
            raise RuntimeError(
                "S3 storage is required. Set USE_S3_STORAGE=true in .env file"
            )
        
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise RuntimeError(
                "AWS credentials are required. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file"
            )
        
        if not settings.AWS_BUCKET_NAME:
            raise RuntimeError(
                "AWS bucket name is required. Set AWS_BUCKET_NAME in .env file"
            )
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            print(f"[S3] Initialized S3 client for bucket: {settings.AWS_BUCKET_NAME}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
    
    def _get_cache_key(self, dataset_id: int, page: int = 1) -> str:
        """Generate Redis cache key for dataset images"""
        return f"dataset:{dataset_id}:images:page:{page}"
    
    def _encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _get_image_from_s3(self, s3_key: str) -> Optional[bytes]:
        """Download image from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=settings.AWS_BUCKET_NAME,
                Key=s3_key
            )
            return response['Body'].read()
        except ClientError as e:
            print(f"[S3] Error downloading {s3_key}: {e}")
            return None
        except Exception as e:
            print(f"[S3] Unexpected error downloading {s3_key}: {e}")
            return None
    
    def _get_image_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """Extract image metadata using PIL"""
        try:
            img = Image.open(io.BytesIO(image_data))
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(image_data)
            }
        except Exception as e:
            print(f"[METADATA] Error extracting metadata: {e}")
            return {
                "size_bytes": len(image_data)
            }
    
    def _process_single_image(self, img_obj: Any) -> Optional[Dict[str, Any]]:
        """
        Process a single image: download from S3, encode to base64, extract metadata
        Used for parallel processing
        
        Args:
            img_obj: Image database object
            
        Returns:
            Processed image dict or None if failed
        """
        try:
            # Get image data from S3
            s3_key = img_obj.file_path.replace("\\", "/")
            image_data = self._get_image_from_s3(s3_key)
            
            if not image_data:
                print(f"[ERROR] Failed to load image from S3: {img_obj.filename} (key: {s3_key})")
                return None
            
            # Encode to base64
            base64_data = self._encode_image_to_base64(image_data)
            
            # Get metadata
            metadata = self._get_image_metadata(image_data)
            
            # Build response object
            return {
                "id": img_obj.id,
                "filename": img_obj.filename,
                "data": base64_data,
                "content_type": f"image/{metadata.get('format', 'jpeg').lower()}",
                "width": metadata.get("width") or img_obj.width,
                "height": metadata.get("height") or img_obj.height,
                "size": metadata.get("size_bytes"),
                "is_annotated": bool(img_obj.is_annotated)
            }
        except Exception as e:
            print(f"[ERROR] Exception processing image {img_obj.filename}: {e}")
            return None
    
    def _download_images_parallel(
        self, 
        images: List[Any], 
        max_workers: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Download and process multiple images in parallel
        
        Args:
            images: List of Image database objects
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple of (encoded_images, failed_filenames)
        """
        encoded_images = []
        failed_images = []
        
        start_time = time.time()
        print(f"[PARALLEL] Starting parallel download of {len(images)} images with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_img = {
                executor.submit(self._process_single_image, img_obj): img_obj 
                for img_obj in images
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_img):
                img_obj = future_to_img[future]
                try:
                    result = future.result()
                    if result:
                        encoded_images.append(result)
                    else:
                        failed_images.append(img_obj.filename)
                except Exception as e:
                    print(f"[ERROR] Future exception for {img_obj.filename}: {e}")
                    failed_images.append(img_obj.filename)
        
        elapsed = time.time() - start_time
        print(f"[PARALLEL] Completed in {elapsed:.2f}s - "
              f"Success: {len(encoded_images)}, Failed: {len(failed_images)}")
        
        return encoded_images, failed_images
    
    async def get_cached_images(
        self, 
        dataset_id: int,
        images_db: List[Any],
        page: int = 1,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Get cached images or load from storage and cache
        
        Args:
            dataset_id: Dataset ID
            images_db: List of Image database objects
            page: Page number (1-based)
            limit: Number of images per page (default from settings)
            
        Returns:
            Dictionary with cached images data
        """
        if limit is None:
            limit = settings.IMAGE_CACHE_PAGE_SIZE
        
        cache_key = self._get_cache_key(dataset_id, page)
        
        # Try to get from cache
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"[CACHE HIT] {cache_key}")
            return {
                "dataset_id": dataset_id,
                "page": page,
                "page_size": limit,
                "cached": True,
                "images": cached_data.get("images", []),
                "total_images": cached_data.get("total_images", 0)
            }
        
        print(f"[CACHE MISS] {cache_key}")
        
        # Paginate images
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_images = images_db[start_idx:end_idx]
        
        # Load images using parallel download (optimized)
        encoded_images, failed_filenames = self._download_images_parallel(
            paginated_images,
            max_workers=min(10, len(paginated_images))  # Use up to 10 workers
        )
        
        # S3에서 이미지를 하나도 가져오지 못한 경우 오류 발생
        if len(paginated_images) > 0 and len(encoded_images) == 0:
            print(f"[ERROR] No images could be loaded from S3 for dataset {dataset_id}")
            print(f"[ERROR] Failed images: {len(failed_filenames)}/{len(paginated_images)}")
            raise Exception(
                f"Failed to load images from S3. "
                f"All {len(paginated_images)} images are missing in S3. "
                f"Please upload images to S3 or migrate existing local images."
            )
        
        # 일부만 실패한 경우 경고 로그
        if failed_filenames:
            print(f"[WARN] {len(failed_filenames)}/{len(paginated_images)} images failed to load from S3")
            print(f"[WARN] Failed images: {', '.join(failed_filenames[:10])}")  # Show first 10
        
        # Prepare cache data
        cache_data = {
            "images": encoded_images,
            "total_images": len(images_db),
            "page": page,
            "page_size": limit
        }
        
        # 성공적으로 로드된 이미지가 있는 경우에만 캐시
        if encoded_images:
            redis_client.set(
                cache_key, 
                cache_data, 
                expire=settings.IMAGE_CACHE_TTL
            )
            print(f"[CACHE SET] {cache_key} - {len(encoded_images)} images cached")
        
        return {
            "dataset_id": dataset_id,
            "page": page,
            "page_size": limit,
            "cached": False,
            "images": encoded_images,
            "total_images": len(images_db)
        }
    
    def invalidate_dataset_cache(self, dataset_id: int):
        """Invalidate all cached pages for a dataset"""
        pattern = f"dataset:{dataset_id}:images:page:*"
        result = redis_client.invalidate_pattern(pattern)
        if result:
            print(f"[CACHE] Invalidated dataset {dataset_id} image cache")
        return result
    
    def get_cache_stats(self, dataset_id: int) -> Dict[str, Any]:
        """Get cache statistics for a dataset"""
        # This would require scanning Redis for all pages
        # For now, return basic info
        return {
            "dataset_id": dataset_id,
            "cache_ttl": settings.IMAGE_CACHE_TTL,
            "page_size": settings.IMAGE_CACHE_PAGE_SIZE
        }


# Global service instance
dataset_cache_service = DatasetCacheService()

