from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.dataset import Dataset, DatasetVersion, ExportJob
from app.schemas.dataset_version import ExportJobCreate, ExportJobResponse
from pathlib import Path

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
    """Create a new export job"""
    # Verify dataset or version exists
    if export_request.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == export_request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    elif export_request.version_id:
        version = db.query(DatasetVersion).filter(DatasetVersion.id == export_request.version_id).first()
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
    else:
        raise HTTPException(status_code=400, detail="Either dataset_id or version_id must be provided")

    # Create export job
    export_job = ExportJob(
        dataset_id=export_request.dataset_id,
        version_id=export_request.version_id,
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
    """Download exported dataset"""
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Export job not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Export job not completed yet")

    if not job.file_path or not Path(job.file_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=job.file_path,
        filename=Path(job.file_path).name,
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

        # Process export
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
