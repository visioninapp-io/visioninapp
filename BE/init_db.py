"""
Database initialization script
Creates all tables and optionally seeds with sample data
"""

from app.core.database import Base, engine, SessionLocal
from app.models import *  # Import all models
import sys


def init_database(drop_existing=False):
    """
    Initialize the database with all tables

    Args:
        drop_existing: If True, drops all existing tables before creating new ones
    """
    print("=" * 50)
    print("DATABASE INITIALIZATION")
    print("=" * 50)

    if drop_existing:
        print("\n[WARNING] Dropping all existing tables...")
        response = input("Are you sure you want to drop all tables? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        Base.metadata.drop_all(bind=engine)
        print("[OK] All tables dropped")

    print("\nCreating database tables...")
    Base.metadata.create_all(bind=engine)
    print("[OK] All tables created successfully")

    # Print created tables
    print("\nCreated tables:")
    for table in Base.metadata.sorted_tables:
        print(f"  - {table.name}")

    print("\n" + "=" * 50)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 50)


def seed_sample_data():
    """Seed database with sample data for testing"""
    from app.models.dataset import Dataset, DatasetStatus, DatasetType
    from app.models.model import Model
    from datetime import datetime

    print("\n" + "=" * 50)
    print("SEEDING SAMPLE DATA")
    print("=" * 50)

    db = SessionLocal()

    try:
        # Check if data already exists
        existing_datasets = db.query(Dataset).count()
        if existing_datasets > 0:
            print(f"\n[WARNING] Database already contains {existing_datasets} dataset(s)")
            response = input("Do you want to add more sample data? (yes/no): ")
            if response.lower() != 'yes':
                print("Skipped seeding.")
                return

        # Create sample datasets
        print("\nCreating sample datasets...")

        sample_datasets = [
            {
                "name": "Factory Defect Detection",
                "description": "Manufacturing defect detection dataset",
                "dataset_type": DatasetType.OBJECT_DETECTION,
                "status": DatasetStatus.READY,
                "total_classes": 2,
                "class_names": ["good", "defect"],
                "class_colors": {"good": "#10B981", "defect": "#EF4444"},
                "created_by": "dev-user-001"
            },
            {
                "name": "Retail Product Recognition",
                "description": "Product detection in retail environment",
                "dataset_type": DatasetType.OBJECT_DETECTION,
                "status": DatasetStatus.CREATED,
                "total_classes": 5,
                "class_names": ["bottle", "can", "box", "bag", "carton"],
                "class_colors": {
                    "bottle": "#3B82F6",
                    "can": "#8B5CF6",
                    "box": "#F59E0B",
                    "bag": "#EC4899",
                    "carton": "#10B981"
                },
                "created_by": "dev-user-001"
            },
            {
                "name": "Safety Equipment Detection",
                "description": "Detect safety equipment on workers",
                "dataset_type": DatasetType.OBJECT_DETECTION,
                "status": DatasetStatus.READY,
                "total_classes": 3,
                "class_names": ["helmet", "vest", "gloves"],
                "class_colors": {
                    "helmet": "#EF4444",
                    "vest": "#F59E0B",
                    "gloves": "#3B82F6"
                },
                "created_by": "dev-user-001"
            }
        ]

        created_datasets = []
        for ds_data in sample_datasets:
            dataset = Dataset(**ds_data)
            db.add(dataset)
            created_datasets.append(dataset)

        db.commit()
        print(f"[OK] Created {len(created_datasets)} sample datasets")

        # Create sample models
        print("\nCreating sample models...")

        sample_models = [
            {
                "name": "YOLOv8n-defect-v1",
                "description": "Fast YOLOv8 nano model for defect detection",
                "architecture": "yolov8n",
                "framework": "ultralytics",
                "version": "1.0.0",
                "file_path": "AI/yolov8n.pt",
                "status": "trained",
                "created_by": "dev-user-001"
            },
            {
                "name": "YOLOv8m-products-v1",
                "description": "Medium YOLOv8 for product recognition",
                "architecture": "yolov8m",
                "framework": "ultralytics",
                "version": "1.0.0",
                "file_path": "AI/yolov8m.pt",
                "status": "trained",
                "created_by": "dev-user-001"
            }
        ]

        created_models = []
        for model_data in sample_models:
            model = Model(**model_data)
            db.add(model)
            created_models.append(model)

        db.commit()
        print(f"[OK] Created {len(created_models)} sample models")

        print("\n" + "=" * 50)
        print("SAMPLE DATA SEEDED SUCCESSFULLY")
        print("=" * 50)

    except Exception as e:
        print(f"\n[ERROR] Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize VisionAI Platform database")
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop all existing tables before creating new ones'
    )
    parser.add_argument(
        '--seed',
        action='store_true',
        help='Seed database with sample data'
    )

    args = parser.parse_args()

    # Initialize database
    init_database(drop_existing=args.drop)

    # Seed data if requested
    if args.seed:
        seed_sample_data()

    print("\n[OK] Database is ready!")
    print("You can now start the server with: uvicorn main:app --reload")
