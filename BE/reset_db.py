"""
Database Reset Script
This script drops all tables and recreates them from scratch
WARNING: This will delete all data in the database!
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import inspect, text
from app.core.database import engine, Base
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def drop_all_tables():
    """Drop all tables in the database"""
    logger.info("[DROP] Dropping all tables...")

    try:
        # Get inspector to check existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if not existing_tables:
            logger.info("No tables found in database")
            return

        logger.info(f"Found {len(existing_tables)} tables: {', '.join(existing_tables)}")

        # Disable foreign key checks
        with engine.connect() as connection:
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            connection.commit()

            # Drop each table
            for table in existing_tables:
                try:
                    logger.info(f"Dropping table: {table}")
                    connection.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
                    connection.commit()
                except Exception as e:
                    logger.error(f"Error dropping table {table}: {e}")

            # Re-enable foreign key checks
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            connection.commit()

        logger.info("[SUCCESS] All tables dropped successfully")

    except Exception as e:
        logger.error(f"[ERROR] Error dropping tables: {e}")
        raise

def create_all_tables():
    """Create all tables defined in models"""
    logger.info("[CREATE] Creating all tables...")

    try:
        # Import all models to ensure they are registered with Base
        from app.models import (
            user, project, dataset, dataset_split,
            label_class, label_ontology_version, asset, model, model_version,
            model_artifact, training, evaluation, deployment
        )

        # Create all tables
        Base.metadata.create_all(bind=engine)

        # Verify tables were created
        inspector = inspect(engine)
        created_tables = inspector.get_table_names()

        logger.info(f"[SUCCESS] Created {len(created_tables)} tables: {', '.join(created_tables)}")

    except Exception as e:
        logger.error(f"[ERROR] Error creating tables: {e}")
        raise

def reset_database():
    """Complete database reset: drop all tables and recreate them"""
    logger.info("=" * 80)
    logger.info("DATABASE RESET STARTED")
    logger.info("=" * 80)

    try:
        # Step 1: Drop all tables
        drop_all_tables()

        # Step 2: Create all tables
        create_all_tables()

        logger.info("=" * 80)
        logger.info("[SUCCESS] DATABASE RESET COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[ERROR] DATABASE RESET FAILED: {e}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    # Confirm action
    print("\n" + "=" * 80)
    print("WARNING: This will DELETE ALL DATA in the database!")
    print("=" * 80)
    print(f"Database: {engine.url.database}")
    print(f"Host: {engine.url.host}")
    print(f"User: {engine.url.username}")
    print("=" * 80)

    confirm = input("\nAre you sure you want to reset the database? (yes/no): ")

    if confirm.lower() == 'yes':
        reset_database()
    else:
        logger.info("Database reset cancelled by user")
        print("[CANCELLED] Database reset cancelled")
