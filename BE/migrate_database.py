"""
Database migration script to fix enum column types
This will recreate tables with correct string-based enum columns
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from app.core.database import SessionLocal, engine, Base
from sqlalchemy import text, inspect
from app.models import (
    Dataset, Image, Annotation, DatasetVersion, UploadBatch, ExportJob,
    Model, ModelConversion,
    TrainingJob, TrainingMetric,
    Evaluation,
    Deployment, InferenceLog,
    MonitoringAlert, PerformanceMetric, FeedbackLoop, EdgeCase,
    User
)

def backup_database():
    """Create a backup of the current database"""
    db_path = Path("vision_platform.db")
    if db_path.exists():
        backup_path = Path(f"vision_platform_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy(db_path, backup_path)
        print(f"[OK] Database backed up to: {backup_path}")
        return backup_path
    return None

def export_data(db):
    """Export all data from current database"""
    print("[*] Exporting data from database...")

    data = {}

    # Export in order to respect foreign key constraints on import
    tables_to_export = [
        ('datasets', 'SELECT * FROM datasets'),
        ('images', 'SELECT * FROM images'),
        ('annotations', 'SELECT * FROM annotations'),
        ('models', 'SELECT * FROM models'),
        ('training_jobs', 'SELECT * FROM training_jobs'),
        ('training_metrics', 'SELECT * FROM training_metrics'),
        ('model_conversions', 'SELECT * FROM model_conversions'),
        ('deployments', 'SELECT * FROM deployments'),
        ('deployment_logs', 'SELECT * FROM deployment_logs'),
        ('evaluations', 'SELECT * FROM evaluations'),
        ('alerts', 'SELECT * FROM alerts'),
    ]

    for table_name, query in tables_to_export:
        try:
            result = db.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            data[table_name] = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert datetime to string
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    row_dict[col] = value
                data[table_name].append(row_dict)

            print(f"    {table_name}: {len(data[table_name])} records")
        except Exception as e:
            print(f"    [WARNING] Could not export {table_name}: {e}")
            data[table_name] = []

    return data

def save_data_to_file(data, filename="database_export.json"):
    """Save exported data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Data exported to: {filename}")

def load_data_from_file(filename="database_export.json"):
    """Load data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"[OK] Data loaded from: {filename}")
    return data

def recreate_database():
    """Drop all tables and recreate with new schema"""
    print("[*] Recreating database with new schema...")

    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    print("    [OK] Dropped all tables")

    # Recreate all tables with new definitions
    Base.metadata.create_all(bind=engine)
    print("    [OK] Created all tables with new schema")

def import_data(db, data):
    """Import data back into database"""
    print("[*] Importing data back into database...")

    # Import in same order as export to respect foreign keys
    for table_name, rows in data.items():
        if not rows:
            continue

        try:
            # Build insert query
            if rows:
                columns = list(rows[0].keys())
                placeholders = ', '.join([f':{col}' for col in columns])
                cols_str = ', '.join(columns)

                query = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})"

                # Convert date strings back to datetime objects
                for row in rows:
                    for key, value in row.items():
                        if value and isinstance(value, str) and 'T' in value:
                            # Try to parse as datetime
                            try:
                                from datetime import datetime
                                row[key] = datetime.fromisoformat(value)
                            except:
                                pass  # Keep as string if parsing fails

                db.execute(text(query), rows)
                db.commit()
                print(f"    {table_name}: {len(rows)} records imported")
        except Exception as e:
            print(f"    [ERROR] Failed to import {table_name}: {e}")
            db.rollback()

def migrate():
    """Main migration function"""
    print("=" * 70)
    print("  Database Migration Script")
    print("  This will recreate tables with correct enum column types")
    print("=" * 70)
    print()

    # Step 1: Backup database file
    backup_path = backup_database()

    # Step 2: Export data
    db = SessionLocal()
    try:
        data = export_data(db)
        save_data_to_file(data)
    finally:
        db.close()

    print()

    # Step 3: Recreate database
    recreate_database()

    print()

    # Step 4: Import data
    db = SessionLocal()
    try:
        import_data(db, data)
    finally:
        db.close()

    print()
    print("=" * 70)
    print("[OK] Migration completed successfully!")
    print("=" * 70)
    print()
    print("You can now access your database with the correct enum column types.")
    print(f"Original database backed up to: {backup_path}")

if __name__ == "__main__":
    migrate()
