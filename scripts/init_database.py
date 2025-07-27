"""
Enhanced Database initialization script for EasyML
Creates all necessary tables, collections, and initial data for hybrid architecture
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from motor.motor_asyncio import AsyncIOMotorClient
import logging

from app.core.config import get_settings
from app.models.sql_models import Base
from app.core.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def init_postgresql():
    """Initialize PostgreSQL database and tables"""
    logger.info("Initializing PostgreSQL database...")
    
    try:
        # Create async engine
        engine = create_async_engine(
            settings.postgres_url,
            echo=True,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ PostgreSQL tables created successfully")
        
        # Close engine
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
        raise


async def init_mongodb():
    """Initialize MongoDB database and collections"""
    logger.info("Initializing MongoDB database...")
    
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.mongo_url)
        db = client[settings.mongo_db_name]
        
        # Create collections with proper indexes
        collections_to_create = [
            "user_sessions",
            "dvc_metadata", 
            "mlflow_runs",
            "audit_logs",
            "project_configs"
        ]
        
        for collection_name in collections_to_create:
            # Create collection
            await db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")
        
        # Create indexes for better performance
        await create_mongodb_indexes(db)
        
        logger.info("✅ MongoDB collections and indexes created successfully")
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MongoDB: {e}")
        raise


async def create_mongodb_indexes(db):
    """Create indexes for MongoDB collections"""
    logger.info("Creating MongoDB indexes...")
    
    try:
        # User sessions indexes
        await db.user_sessions.create_index("user_id")
        await db.user_sessions.create_index("session_token")
        await db.user_sessions.create_index("expires_at")
        
        # DVC metadata indexes
        await db.dvc_metadata.create_index([("user_id", 1), ("project_id", 1)])
        await db.dvc_metadata.create_index([("user_id", 1), ("project_id", 1), ("name", 1)])
        await db.dvc_metadata.create_index([("user_id", 1), ("project_id", 1), ("data_type", 1)])
        await db.dvc_metadata.create_index("created_at")
        
        # MLflow runs indexes
        await db.mlflow_runs.create_index([("user_id", 1), ("project_id", 1)])
        await db.mlflow_runs.create_index("experiment_id")
        await db.mlflow_runs.create_index("run_id")
        await db.mlflow_runs.create_index("created_at")
        
        # Audit logs indexes
        await db.audit_logs.create_index([("user_id", 1), ("project_id", 1)])
        await db.audit_logs.create_index("action")
        await db.audit_logs.create_index("timestamp")
        
        # Project configs indexes
        await db.project_configs.create_index([("user_id", 1), ("project_id", 1)])
        await db.project_configs.create_index("config_type")
        
        logger.info("✅ All MongoDB indexes created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create MongoDB indexes: {e}")
        raise


async def init_dvc():
    """Initialize DVC configuration with Azure Blob Storage"""
    logger.info("Initializing DVC configuration...")
    
    try:
        # Check if DVC is already initialized
        if os.path.exists('.dvc'):
            logger.info("DVC already initialized")
        else:
            # Initialize DVC
            import subprocess
            result = subprocess.run(['dvc', 'init'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"DVC init failed: {result.stderr}")
            logger.info("✅ DVC initialized successfully")
        
        # Configure Azure Blob Storage remote if URLs are provided
        if settings.dvc_remote_url and settings.dvc_azure_connection_string:
            import subprocess
            
            # Remove existing remote if it exists
            subprocess.run(['dvc', 'remote', 'remove', settings.dvc_remote_name], 
                         capture_output=True, text=True)
            
            # Add Azure remote
            result = subprocess.run([
                'dvc', 'remote', 'add', '-d', settings.dvc_remote_name, settings.dvc_remote_url
            ], capture_output=True, text=True)
            
            if result.returncode != 0 and "already exists" not in result.stderr:
                raise Exception(f"DVC remote add failed: {result.stderr}")
            
            # Configure Azure connection string
            result = subprocess.run([
                'dvc', 'remote', 'modify', settings.dvc_remote_name, 
                'connection_string', settings.dvc_azure_connection_string
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"DVC Azure configuration failed: {result.stderr}")
            
            logger.info(f"✅ DVC Azure remote configured: {settings.dvc_remote_url}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize DVC: {e}")
        raise


async def create_default_admin_user():
    """Create a default admin user if none exists"""
    logger.info("Creating default admin user...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.init_databases()
        
        from app.models.sql_models import User
        from app.core.auth import get_password_hash
        
        # Check if any users exist
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(User))
            existing_users = result.scalars().all()
            
            if not existing_users:
                # Create default admin user
                admin_user = User(
                    username="admin",
                    email="admin@easyml.com",
                    hashed_password=get_password_hash("admin123"),
                    is_active=True,
                    is_superuser=True
                )
                
                session.add(admin_user)
                await session.commit()
                
                logger.info("✅ Default admin user created (username: admin, password: admin123)")
            else:
                logger.info("Users already exist, skipping default admin creation")
        
        await db_manager.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to create default admin user: {e}")
        raise


def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "datasets", 
        "uploads",
        "logs",
        "dvc_storage"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


async def main():
    """Main initialization function"""
    logger.info("🚀 Starting EasyML database initialization...")
    
    try:
        # Create directories
        logger.info("1. Creating directories...")
        create_directories()
        
        # Initialize PostgreSQL
        logger.info("2. Initializing PostgreSQL...")
        await init_postgresql()
        
        # Initialize MongoDB
        logger.info("3. Initializing MongoDB...")
        await init_mongodb()
        
        # Initialize DVC
        logger.info("4. Initializing DVC...")
        await init_dvc()
        
        # Create default admin user
        logger.info("5. Creating default admin user...")
        await create_default_admin_user()
        
        logger.info("🎉 EasyML database initialization completed successfully!")
        logger.info("")
        logger.info("📋 Summary:")
        logger.info("  ✅ PostgreSQL tables created")
        logger.info("  ✅ MongoDB collections and indexes created")
        logger.info("  ✅ DVC initialized and configured")
        logger.info("  ✅ Default admin user created")
        logger.info("  ✅ Necessary directories created")
        logger.info("")
        logger.info("🔐 Default Admin Credentials:")
        logger.info("  Username: admin")
        logger.info("  Password: admin123")
        logger.info("  Email: admin@easyml.com")
        logger.info("")
        logger.info("⚠️  Please change the admin password after first login!")
        
    except Exception as e:
        logger.error(f"💥 Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
