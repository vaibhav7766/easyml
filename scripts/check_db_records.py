#!/usr/bin/env python3
"""
Quick script to check database records
"""
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import engine, SessionLocal
from app.models.sql_models import ModelVersion, Project, User

try:
    print("🔍 Checking database records...")
    print(f"🔧 Database URL: {engine.url}")
    
    # Check if there are any database files (SQLite)
    db_files = list(Path('.').glob('*.db')) + list(Path('.').glob('*.sqlite*'))
    if db_files:
        print(f"� Found database files: {db_files}")
    
    with SessionLocal() as db:
        # Check ModelVersion records
        models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(5).all()
        print(f"\n📊 Found {len(models)} ModelVersion records:")
        for model in models:
            print(f"  - ID: {model.id}")
            print(f"    Name: {model.name}")
            print(f"    Version: {model.version}")
            print(f"    Type: {model.model_type}")
            print(f"    Status: {model.status}")
            print(f"    Project ID: {model.project_id}")
            print(f"    Created: {model.created_at}")
            print()
        
        # Check recent projects
        projects = db.query(Project).order_by(Project.created_at.desc()).limit(3).all()
        print(f"📁 Found {len(projects)} recent projects:")
        for project in projects:
            print(f"  - ID: {project.id}")
            print(f"    Name: {project.name}")
            print(f"    Owner ID: {project.owner_id}")
            print(f"    Created: {project.created_at}")
            print()
        
        # Check recent users  
        users = db.query(User).order_by(User.created_at.desc()).limit(3).all()
        print(f"👤 Found {len(users)} recent users:")
        for user in users:
            print(f"  - ID: {user.id}")
            print(f"    Username: {user.username}")
            print(f"    Created: {user.created_at}")
            print()
        
except Exception as e:
    print(f"❌ Database connection error: {e}")
