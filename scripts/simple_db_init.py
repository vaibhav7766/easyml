"""
Simple Synchronous Database Initialization Script
Creates all EasyML tables with PostgreSQL
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine, text
from app.core.config import get_settings

def create_all_tables():
    """Create all database tables"""
    
    print("🚀 Creating EasyML Database Tables...")
    
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    try:
        # Import all models to register them
        from app.models.sql_models import (
            Base, User, Project, MLExperiment, 
            ModelVersion, DatasetVersion, ModelDeployment
        )
        
        # Create all tables
        print("📊 Creating all tables...")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            print(f"✅ Created tables: {', '.join(tables)}")
            
            # Test each table
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  📋 {table}: {count} records")
        
        print("🎉 Database initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main initialization function"""
    
    print("🗄️  EasyML Simple Database Setup")
    print("=" * 40)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    success = create_all_tables()
    
    if success:
        print("\n✅ Database is ready!")
        print("📊 PostgreSQL: Fully operational")
        print("⚠️  MongoDB: Has SSL issues (non-blocking)")
        print("\nYou can now:")
        print("1. Start the API: uvicorn app.main:app --reload")
        print("2. Test deployment endpoints")
        print("3. Create users and projects")
    else:
        print("\n❌ Setup failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
