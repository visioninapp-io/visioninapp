"""
Fix invalid 'TRAINED' status in models table
Run this once to fix existing database records
"""

from app.core.database import SessionLocal, engine
from app.models.model import Model, ModelStatus
from sqlalchemy import text

def fix_model_status():
    """Fix models with invalid 'TRAINED' status"""
    db = SessionLocal()

    try:
        print("🔍 Checking for models with invalid status...")

        # Direct SQL query to find models with 'TRAINED' status
        result = db.execute(text("SELECT id, name, status FROM models WHERE status = 'TRAINED'"))
        invalid_models = result.fetchall()

        if not invalid_models:
            print("✅ No invalid model status found!")
            return

        print(f"⚠️  Found {len(invalid_models)} models with 'TRAINED' status")

        for model_id, name, status in invalid_models:
            print(f"   - Model {model_id}: {name} (status: {status})")

        # Update to COMPLETED
        print("\n🔧 Updating to COMPLETED status...")
        db.execute(text("UPDATE models SET status = 'completed' WHERE status = 'TRAINED'"))
        db.commit()

        print("✅ Successfully updated all models!")

        # Verify
        result = db.execute(text("SELECT id, name, status FROM models"))
        all_models = result.fetchall()

        print(f"\n📊 Current models in database:")
        for model_id, name, status in all_models:
            print(f"   - Model {model_id}: {name} (status: {status})")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("="*60)
    print("  Fix Model Status Script")
    print("="*60)
    print()

    fix_model_status()

    print()
    print("="*60)
    print("  Done!")
    print("="*60)
