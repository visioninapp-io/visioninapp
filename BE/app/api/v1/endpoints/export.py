from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.dataset import Dataset, DatasetVersion, ExportJob
from app.schemas.dataset_version import ExportJobCreate, ExportJobResponse
from pathlib import Path
import urllib.parse

router = APIRouter()


@router.get("/", response_model=List[ExportJobResponse])
async def get_export_jobs(
    skip: int = 0,
    limit: int = 10000,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all export jobs for current user"""
    # For now, return all export jobs (can filter by created_by later)
    jobs = db.query(ExportJob).offset(skip).limit(limit).all()
    return jobs


@router.post("/", response_model=ExportJobResponse, status_code=status.HTTP_201_CREATED)
async def create_export_job(
    export_request: ExportJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new export job for dataset or model"""
    # Verify dataset, version, or model exists
    if export_request.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == export_request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    elif export_request.version_id:
        version = db.query(DatasetVersion).filter(DatasetVersion.id == export_request.version_id).first()
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
    elif export_request.model_id:
        from app.models.model import Model
        model = db.query(Model).filter(Model.id == export_request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
    else:
        raise HTTPException(status_code=400, detail="Either dataset_id, version_id, or model_id must be provided")

    # Create export job
    export_job = ExportJob(
        dataset_id=export_request.dataset_id,
        version_id=export_request.version_id,
        model_id=export_request.model_id,
        export_format="zip",
        include_images=int(export_request.include_images),
        status="pending",
        created_by=current_user["uid"]
    )

    db.add(export_job)
    db.commit()
    db.refresh(export_job)

    # Start background task to process export
    background_tasks.add_task(process_export_job, export_job.id)

    return export_job


@router.get("/{export_id}", response_model=ExportJobResponse)
async def get_export_job(
    export_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get specific export job"""
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Export job not found")

    return job


@router.get("/{export_id}/download")
async def download_export(
    export_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Download exported dataset or model"""
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Export job not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Export job not completed yet")

    if not job.file_path or not Path(job.file_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    # Get dataset or model name for better filename
    filename = Path(job.file_path).name  # Default to existing filename
    
    # Debug: Log the actual file path
    print(f"[Export Download] Export ID: {export_id}")
    print(f"[Export Download] File path: {job.file_path}")
    print(f"[Export Download] Filename from path: {filename}")
    print(f"[Export Download] Dataset ID: {job.dataset_id}, Model ID: {job.model_id}, Version ID: {job.version_id}")
    
    # If filename is in export_{id}.zip format, reconstruct it with dataset/model name
    # Also check if filename matches export_{id}.zip pattern (with numbers)
    import re
    export_pattern = re.match(r'^export_(\d+)\.zip$', filename)
    if export_pattern or (filename.startswith('export_') and filename.endswith('.zip')):
        # Extract timestamp from job creation or completion time
        from datetime import datetime
        if job.completed_at:
            timestamp = job.completed_at.strftime("%Y%m%d_%H%M%S")
        elif job.created_at:
            timestamp = job.created_at.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try to get model name first
        if job.model_id:
            from app.models.model import Model
            model = db.query(Model).filter(Model.id == job.model_id).first()
            if model and model.name:
                # Sanitize model name for filename
                safe_name = "".join(c for c in model.name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_name = safe_name.replace(' ', '_')
                filename = f"model_{safe_name}_{timestamp}.zip"
                print(f"[Export Download] Reconstructed filename from model: {filename}")
        # Try dataset_id
        elif job.dataset_id:
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if dataset and dataset.name:
                # Sanitize dataset name for filename
                safe_name = "".join(c for c in dataset.name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_name = safe_name.replace(' ', '_')
                filename = f"dataset_{safe_name}_{timestamp}.zip"
                print(f"[Export Download] Reconstructed filename from dataset: {filename}")
        # Try version_id (which can lead to dataset)
        elif job.version_id:
            version = db.query(DatasetVersion).filter(DatasetVersion.id == job.version_id).first()
            if version and version.dataset_id:
                dataset = db.query(Dataset).filter(Dataset.id == version.dataset_id).first()
                if dataset and dataset.name:
                    # Sanitize dataset name for filename
                    safe_name = "".join(c for c in dataset.name if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_name = safe_name.replace(' ', '_')
                    filename = f"dataset_{safe_name}_{timestamp}.zip"
                    print(f"[Export Download] Reconstructed filename from version->dataset: {filename}")
        
        # Debug logging
        print(f"[Export Download] Export ID: {export_id}, Dataset ID: {job.dataset_id}, Model ID: {job.model_id}, Version ID: {job.version_id}")
        print(f"[Export Download] Original filename: {Path(job.file_path).name}, New filename: {filename}")
    
    # Set Content-Disposition header with proper filename
    # URL encode the filename for safe transmission
    encoded_filename = urllib.parse.quote(filename.encode('utf-8'))
    
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}',
        'Content-Type': 'application/zip'
    }
    
    # Use FileResponse for better memory efficiency (streaming for large files)
    return FileResponse(
        path=job.file_path,
        filename=filename,
        headers=headers,
        media_type="application/zip"
    )


@router.delete("/{export_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_export_job(
    export_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete export job and its file"""
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Export job not found")

    # Delete file if exists
    if job.file_path and Path(job.file_path).exists():
        Path(job.file_path).unlink()

    db.delete(job)
    db.commit()
    return None


def process_export_job(export_id: int):
    """Background task to process export job"""
    from app.services.export_service import ExportService
    from app.core.database import SessionLocal

    db = SessionLocal()

    try:
        job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
        if not job:
            return

        job.status = "processing"
        db.commit()

        # Get export service
        export_service = ExportService()

        # Process export based on job type
        if job.model_id:
            # Export model
            result = export_service.export_model(
                export_id=job.id,
                model_id=job.model_id,
                db=db
            )
        else:
            # Export dataset
            result = export_service.export_dataset(
                export_id=job.id,
                dataset_id=job.dataset_id,
                version_id=job.version_id,
                include_images=bool(job.include_images),
                db=db
            )

        # Update job with results
        job.file_path = result['file_path']
        job.file_size = result['file_size']
        job.download_url = f"/api/v1/export/{job.id}/download"
        job.status = "completed"

        from app.utils.timezone import get_kst_now_naive
        job.completed_at = get_kst_now_naive()

        db.commit()

    except Exception as e:
        if 'job' in locals() and job is not None:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        print(f"Error processing export: {e}")

    finally:
        db.close()
