"""
Fix enum values in database to use lowercase values instead of uppercase names
This ensures compatibility with the updated model definitions
"""

from app.core.database import SessionLocal, engine
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

def fix_enum_values():
    """Fix enum columns to use string values instead of native enums"""
    db = SessionLocal()

    try:
        print("=" * 70)
        print("  Database Enum Fix Script")
        print("=" * 70)
        print()

        # Check if we're using SQLite or PostgreSQL
        inspector = inspect(engine)
        dialect_name = engine.dialect.name
        print(f"Database dialect: {dialect_name}")
        print()

        # Fix models table
        print("[*] Checking models table...")
        result = db.execute(text("SELECT id, name, status, framework FROM models"))
        models = result.fetchall()

        if models:
            print(f"    Found {len(models)} models")

            # Map potential uppercase enum names to lowercase values
            status_map = {
                'TRAINING': 'training',
                'COMPLETED': 'completed',
                'TRAINED': 'completed',  # Legacy value
                'FAILED': 'failed',
                'CONVERTING': 'converting',
                'READY': 'ready'
            }

            framework_map = {
                'PYTORCH': 'pytorch',
                'TENSORFLOW': 'tensorflow',
                'ONNX': 'onnx',
                'TENSORRT': 'tensorrt',
                'OPENVINO': 'openvino',
                'COREML': 'coreml'
            }

            updates_made = 0
            for model_id, name, status, framework in models:
                new_status = status_map.get(status, status.lower() if status else 'training')
                new_framework = framework_map.get(framework, framework.lower() if framework else 'pytorch')

                if status != new_status or framework != new_framework:
                    print(f"   - Updating Model {model_id} ({name})")
                    print(f"     Status: {status} -> {new_status}")
                    print(f"     Framework: {framework} -> {new_framework}")

                    db.execute(
                        text("UPDATE models SET status = :status, framework = :framework WHERE id = :id"),
                        {"id": model_id, "status": new_status, "framework": new_framework}
                    )
                    updates_made += 1

            if updates_made > 0:
                db.commit()
                print(f"    [OK] Updated {updates_made} model(s)")
            else:
                print(f"    [OK] All models already have correct values")
        else:
            print("    No models found")

        print()

        # Fix training_jobs table
        print("[*] Checking training_jobs table...")
        result = db.execute(text("SELECT id, name, status FROM training_jobs"))
        jobs = result.fetchall()

        if jobs:
            print(f"   Found {len(jobs)} training jobs")

            status_map = {
                'PENDING': 'pending',
                'RUNNING': 'running',
                'PAUSED': 'paused',
                'COMPLETED': 'completed',
                'FAILED': 'failed',
                'CANCELLED': 'cancelled'
            }

            updates_made = 0
            for job_id, name, status in jobs:
                new_status = status_map.get(status, status.lower() if status else 'pending')

                if status != new_status:
                    print(f"   - Updating Job {job_id} ({name}): {status} -> {new_status}")

                    db.execute(
                        text("UPDATE training_jobs SET status = :status WHERE id = :id"),
                        {"id": job_id, "status": new_status}
                    )
                    updates_made += 1

            if updates_made > 0:
                db.commit()
                print(f"    [OK] Updated {updates_made} job(s)")
            else:
                print(f"    [OK] All jobs already have correct values")
        else:
            print("    No training jobs found")

        print()

        # Verify training metrics exist
        print("[*] Checking training_metrics table...")
        result = db.execute(text("""
            SELECT
                tm.id,
                tm.training_job_id,
                tm.epoch,
                tm.train_loss,
                tm.train_accuracy,
                tj.name as job_name
            FROM training_metrics tm
            LEFT JOIN training_jobs tj ON tm.training_job_id = tj.id
            ORDER BY tm.training_job_id, tm.epoch
        """))
        metrics = result.fetchall()

        if metrics:
            print(f"    Found {len(metrics)} metric records")

            # Group by job
            jobs_with_metrics = {}
            for metric in metrics:
                job_id = metric[1]
                job_name = metric[5] or f"Job {job_id}"
                if job_id not in jobs_with_metrics:
                    jobs_with_metrics[job_id] = {"name": job_name, "count": 0}
                jobs_with_metrics[job_id]["count"] += 1

            print(f"\n    [*] Metrics by training job:")
            for job_id, info in jobs_with_metrics.items():
                print(f"        - {info['name']}: {info['count']} epochs")
        else:
            print("    [WARNING] No training metrics found!")
            print("    This explains why the metrics chart is empty")

        print()
        print("=" * 70)
        print("  Summary")
        print("=" * 70)

        # Final verification
        result = db.execute(text("SELECT COUNT(*) as count FROM models"))
        model_count = result.fetchone()[0]

        result = db.execute(text("SELECT COUNT(*) as count FROM training_jobs"))
        job_count = result.fetchone()[0]

        result = db.execute(text("SELECT COUNT(*) as count FROM training_metrics"))
        metric_count = result.fetchone()[0]

        print(f"Models: {model_count}")
        print(f"Training Jobs: {job_count}")
        print(f"Training Metrics: {metric_count}")
        print()

        if metric_count == 0:
            print("[WARNING] No training metrics found in database!")
            print("          The metrics may not have been saved during training.")
            print("          You may need to restart training to populate metrics.")

        print("=" * 70)
        print("[OK] Database enum fix completed!")
        print("=" * 70)

    except SQLAlchemyError as e:
        print(f"\n[ERROR] Database error: {e}")
        db.rollback()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    fix_enum_values()
