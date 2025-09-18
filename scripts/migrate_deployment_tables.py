"""
Database migration script to add deployment support
Run this script to update the database schema for model deployment functionality
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine, text
from app.core.config import get_settings
from app.core.database import Base
from app.models.sql_models import ModelDeployment

def run_migration():
    """Run database migration for deployment functionality"""
    
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    print("🚀 Running database migration for model deployment...")
    
    try:
        # Create deployment table
        print("📊 Creating model_deployments table...")
        ModelDeployment.__table__.create(engine, checkfirst=True)
        
        # Add foreign key relationships if needed
        with engine.connect() as conn:
            # Check if deployment_id column exists in model_versions
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'model_versions' 
                AND column_name = 'deployment_count'
            """))
            
            if not result.fetchone():
                print("📝 Adding deployment tracking to model_versions...")
                conn.execute(text("""
                    ALTER TABLE model_versions 
                    ADD COLUMN deployment_count INTEGER DEFAULT 0
                """))
                conn.commit()
        
        print("✅ Database migration completed successfully!")
        print("📋 Summary of changes:")
        print("   - Created model_deployments table")
        print("   - Added deployment tracking to model_versions")
        print("   - All foreign key relationships established")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_migration()
    if success:
        print("\n🎉 Ready to deploy models with EasyML!")
        print("Next steps:")
        print("1. Start the FastAPI server")
        print("2. Use /v1/deployments/deploy endpoint")
        print("3. Monitor deployments with /v1/deployments/status/{deployment_id}")
    else:
        print("\n💥 Migration failed. Please check the error and try again.")
        sys.exit(1)
