import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
import boto3


class ExportService:
    """Service for exporting datasets (ZIP only, S3 기반)"""

    def __init__(self):
        # ZIP 파일 저장되는 경로
        self.export_dir = Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_dataset(
        self,
        export_id: int,
        dataset_id: Optional[int],
        version_id: Optional[int],
        include_images: bool,
        db: Session
    ) -> Dict[str, Any]:
        """
        Export dataset by zipping the entire S3 prefix.
        """
        from app.models.dataset import Dataset, DatasetVersion
        from app.core.config import settings

        s3 = boto3.client("s3")
        bucket = settings.AWS_BUCKET_NAME

        # -------------------------------------------------
        # 1) dataset_id resolve from version_id if needed
        # -------------------------------------------------
        if version_id:
            version = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.id == version_id)
                .first()
            )
            if not version:
                raise ValueError(f"Version {version_id} not found")
            dataset_id = version.dataset_id

        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # -------------------------------------------------
        # 2) Determine S3 prefix (dataset.name 우선)
        # -------------------------------------------------
        name_prefix = f"datasets/{dataset.name}/"
        id_prefix = f"datasets/{dataset.id}/"

        def prefix_exists(prefix: str) -> bool:
            resp = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1
            )
            return "Contents" in resp

        if prefix_exists(name_prefix):
            prefix = name_prefix
        elif prefix_exists(id_prefix):
            prefix = id_prefix
        else:
            raise ValueError(
                f"S3 prefix not found for dataset: "
                f"{name_prefix} or {id_prefix}"
            )

        # -------------------------------------------------
        # 3) Create temp export directory
        # -------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{dataset.name}_{timestamp}"

        temp_dir = self.export_dir / export_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 4) Download entire S3 prefix → local
        # -------------------------------------------------
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                # Remove prefix to create relative path structure
                rel_path = key[len(prefix):]
                local_path = temp_dir / rel_path

                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))

        # -------------------------------------------------
        # 5) ZIP creation
        # -------------------------------------------------
        zip_path = self.export_dir / f"{export_name}.zip"
        self._create_zip(temp_dir, zip_path)

        # temp directory 삭제
        shutil.rmtree(temp_dir)

        return {
            "file_path": str(zip_path),
            "file_size": zip_path.stat().st_size
        }

    @staticmethod
    def _create_zip(source_dir: Path, output_zip: Path):
        """Directory → ZIP"""
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    full_path = Path(root) / file
                    arcname = full_path.relative_to(source_dir)
                    zipf.write(full_path, str(arcname))