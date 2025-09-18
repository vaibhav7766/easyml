"""
Database Connection Test Script
Tests PostgreSQL and MongoDB connections with updated credentials
"""
import os
import sys
import asyncio
from pathlib import Path
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.core.config import get_settings

def test_postgresql_connection():
    """Test PostgreSQL connection"""
    print("🔍 Testing PostgreSQL Connection...")
    
    try:
        settings = get_settings()
        print(f"📡 Connecting to: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'Azure PostgreSQL'}")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        # Test connection
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL Connected Successfully!")
            print(f"📊 Database Version: {version}")
            
            # Test database name
            result = conn.execute(text("SELECT current_database();"))
            db_name = result.fetchone()[0]
            print(f"🗄️  Current Database: {db_name}")
            
            # Check if our tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'projects', 'model_versions')
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            
            if tables:
                print(f"📋 Existing EasyML Tables: {', '.join(tables)}")
            else:
                print("📋 No EasyML tables found (run migrations to create them)")
            
            # Test write permission
            try:
                conn.execute(text("CREATE TEMP TABLE test_table (id INTEGER);"))
                conn.execute(text("DROP TABLE test_table;"))
                print("✅ Write Permission: OK")
            except Exception as e:
                print(f"⚠️  Write Permission: Limited - {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL Connection Failed: {e}")
        return False

def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB Connection...")
    
    try:
        settings = get_settings()
        print(f"📡 Connecting to: MongoDB Atlas")
        
        # Create MongoDB client
        client = MongoClient(settings.mongodb_url)
        
        # Test connection with ping
        client.admin.command('ping')
        print("✅ MongoDB Connected Successfully!")
        
        # Get database
        db = client[settings.mongodb_db_name]
        print(f"🗄️  Database: {settings.mongodb_db_name}")
        
        # Test collections
        collections = db.list_collection_names()
        if collections:
            print(f"📋 Existing Collections: {', '.join(collections)}")
        else:
            print("📋 No collections found")
        
        # Test write permission
        try:
            test_collection = db.test_connection
            test_doc = {
                "test": True,
                "timestamp": datetime.utcnow(),
                "message": "Connection test successful"
            }
            result = test_collection.insert_one(test_doc)
            print(f"✅ Write Permission: OK (Document ID: {result.inserted_id})")
            
            # Clean up test document
            test_collection.delete_one({"_id": result.inserted_id})
            
        except Exception as e:
            print(f"⚠️  Write Permission: Limited - {e}")
        
        # Get server info
        server_info = client.server_info()
        print(f"📊 MongoDB Version: {server_info['version']}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return False

def test_database_integration():
    """Test database integration with EasyML models"""
    print("\n🔍 Testing EasyML Database Integration...")
    
    try:
        from app.core.database import get_db, get_database_db
        from app.models.sql_models import Base
        from sqlalchemy import create_engine
        
        settings = get_settings()
        
        # Test SQLAlchemy models
        engine = create_engine(settings.database_url)
        
        # Check if we can import all models
        from app.models.sql_models import User, Project, ModelVersion, MLExperiment, DatasetVersion
        print("✅ SQLAlchemy Models: Imported Successfully")
        
        # Test MongoDB schemas
        from app.models.mongo_schemas import MLFlowRunDocument
        print("✅ MongoDB Schemas: Imported Successfully")
        
        # Test database dependencies
        db_gen = get_db()
        print("✅ Database Dependency: Ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Database Integration Test Failed: {e}")
        return False

def main():
    """Run all database connection tests"""
    print("🧪 EasyML Database Connection Test")
    print("=" * 50)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    results = {
        "postgresql": test_postgresql_connection(),
        "mongodb": test_mongodb_connection(),
        "integration": test_database_integration()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("-" * 25)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.upper():12} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All database connections are working!")
        print("🚀 EasyML is ready to use with your updated database URLs!")
        print("\nNext steps:")
        print("1. Run migrations: python scripts/migrate_deployment_tables.py")
        print("2. Start the FastAPI server: uvicorn app.main:app --reload")
        print("3. Test the deployment endpoints")
    else:
        print("\n⚠️  Some database connections failed.")
        print("Please check your credentials and network connectivity.")
        
        # Provide specific guidance
        if not results["postgresql"]:
            print("\nPostgreSQL Issues:")
            print("- Verify the connection string format")
            print("- Check if the Azure PostgreSQL server allows your IP")
            print("- Confirm username/password are correct")
            
        if not results["mongodb"]:
            print("\nMongoDB Issues:")
            print("- Verify the MongoDB Atlas connection string")
            print("- Check if your IP is whitelisted in Atlas")
            print("- Confirm username/password are correct")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
