#!/usr/bin/env python3
"""
Check ModelVersion records in the database
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the app directory to Python path
sys.path.insert(0, '/mnt/Project/Projects/easyml')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.sql_models import ModelVersion, Project, User
from app.core.database import Base
import uuid
from app.core.config import settings

def main():
    # Get database URL from environment
    postgres_url = settings.postgres_url
    print(f"🔍 Database URL: {postgres_url[:50]}...")
    
    try:
        # Create engine with SSL support for Azure
        engine = create_engine(
            postgres_url,
            connect_args={
                "sslmode": "require",
                "sslcert": None,
                "sslkey": None,
                "sslrootcert": None
            }
        )
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("✅ Connected to database successfully!")
        
        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        print(f"📊 Available tables: {table_names}")
        
        # Count total ModelVersion records
        try:
            total_models = session.query(ModelVersion).count()
            print(f"📈 Total ModelVersion records: {total_models}")
            
            # Get recent ModelVersion records
            recent_models = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(5).all()
            
            if recent_models:
                # Get recent records with DVC fields
                print("\n🔍 Recent ModelVersion records with DVC info:")
                for model in recent_models:
                    print(f"  - ID: {model.id}")
                    print(f"    Name: {model.name}")
                    print(f"    Project ID: {model.project_id}")
                    print(f"    Status: {model.status}")
                    print(f"    MLflow Run ID: {model.mlflow_run_id or 'None'}")
                    print(f"    DVC Path: {model.dvc_path or 'None'}")
                    print(f"    MLflow Model URI: {model.mlflow_model_uri or 'None'}")
                    print(f"    Storage Path: {model.storage_path}")
                    print(f"    Created: {model.created_at}")
                    print()
                    print("    ---")
            else:
                print("❌ No ModelVersion records found!")
                
        except Exception as e:
            print(f"❌ Error querying ModelVersion: {e}")
            
        # Count projects
        try:
            total_projects = session.query(Project).count()
            print(f"📊 Total Project records: {total_projects}")
            
            # Get recent projects
            recent_projects = session.query(Project).order_by(Project.created_at.desc()).limit(3).all()
            
            if recent_projects:
                print("\n🔍 Recent Project records:")
                for project in recent_projects:
                    print(f"  - ID: {project.id}")
                    print(f"    Name: {project.name}")
                    print(f"    Owner ID: {project.owner_id}")
                    print(f"    Created: {project.created_at}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error querying Projects: {e}")
            
        # Count users
        try:
            total_users = session.query(User).count()
            print(f"👥 Total User records: {total_users}")
        except Exception as e:
            print(f"❌ Error querying Users: {e}")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
