#!/usr/bin/env python3
"""
Add mlflow_run_id column to model_versions table
"""
import os
import psycopg2
from sqlalchemy import text
from dotenv import load_dotenv
from app.core.database import engine

# Load environment variables from .env file
load_dotenv()

def add_mlflow_run_id_column():
    """Add mlflow_run_id column to model_versions table"""
    try:
        with engine.connect() as conn:
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='model_versions' AND column_name='mlflow_run_id'
            """))
            
            if result.fetchone() is None:
                print("Adding mlflow_run_id column to model_versions table...")
                conn.execute(text("""
                    ALTER TABLE model_versions 
                    ADD COLUMN mlflow_run_id VARCHAR(100)
                """))
                
                # Add index
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_model_versions_mlflow_run_id 
                    ON model_versions(mlflow_run_id)
                """))
                
                conn.commit()
                print("✅ Column added successfully!")
            else:
                print("✅ Column already exists!")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_mlflow_run_id_column()
