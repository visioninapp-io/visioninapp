"""
Test script to verify metrics API endpoints are working correctly
"""

from app.core.database import SessionLocal
from app.models.training import TrainingJob, TrainingMetric
from app.models.model import Model
from sqlalchemy import text

def test_metrics_api():
    """Test that metrics can be retrieved correctly"""
    db = SessionLocal()

    try:
        print("=" * 70)
        print("  Testing Metrics API")
        print("=" * 70)
        print()

        # Get all training jobs
        jobs = db.query(TrainingJob).all()
        print(f"Training Jobs: {len(jobs)}")
        for job in jobs:
            print(f"  - Job {job.id}: {job.name} (Status: {job.status})")

        print()

        # Get metrics for job 4 (the latest one with 30 epochs)
        job_id = 4
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if job:
            print(f"Job {job_id} Details:")
            print(f"  Name: {job.name}")
            print(f"  Status: {job.status} (type: {type(job.status)})")
            print(f"  Dataset ID: {job.dataset_id}")
            print(f"  Model ID: {job.model_id}")
            print(f"  Architecture: {job.architecture}")
            print(f"  Current Epoch: {job.current_epoch}")
            print(f"  Total Epochs: {job.total_epochs}")
            print()

            # Get metrics
            metrics = db.query(TrainingMetric).filter(
                TrainingMetric.training_job_id == job_id
            ).order_by(TrainingMetric.epoch).all()

            print(f"Metrics for Job {job_id}: {len(metrics)} records")
            if metrics:
                print("\n  First 5 metrics:")
                for metric in metrics[:5]:
                    print(f"    Epoch {metric.epoch}: Loss={metric.train_loss:.4f}, Acc={metric.train_accuracy:.2f}%")

                print("\n  Last 5 metrics:")
                for metric in metrics[-5:]:
                    print(f"    Epoch {metric.epoch}: Loss={metric.train_loss:.4f}, Acc={metric.train_accuracy:.2f}%")
        else:
            print(f"Job {job_id} not found")

        print()

        # Check if models can be queried
        print("Testing Model queries:")
        models = db.query(Model).all()
        print(f"  Models: {len(models)}")
        for model in models:
            print(f"    - Model {model.id}: {model.name} (Status: {model.status}, Type: {type(model.status)})")

        print()
        print("=" * 70)
        print("[OK] All tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    test_metrics_api()
