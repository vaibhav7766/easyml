"""
Add 'tag' column to existing DatasetVersion table
Run this script if you have existing data in the DatasetVersion table
"""
from sqlalchemy import text
from app.core.database import engine

def add_tag_column():
    """Add tag column to DatasetVersion table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'dataset_versions' 
                AND column_name = 'tag'
            """))
            
            if not result.fetchone():
                # Add the tag column with default value
                conn.execute(text("""
                    ALTER TABLE dataset_versions 
                    ADD COLUMN tag VARCHAR(100) NOT NULL DEFAULT 'raw data'
                """))
                
                # Update existing records based on version pattern
                conn.execute(text("""
                    UPDATE dataset_versions 
                    SET tag = CASE 
                        WHEN version = 'V1' THEN 'raw data'
                        ELSE 'preprocessed data'
                    END
                """))
                
                conn.commit()
                print("✅ Successfully added 'tag' column to dataset_versions table")
            else:
                print("ℹ️ 'tag' column already exists in dataset_versions table")
                
    except Exception as e:
        print(f"❌ Error adding tag column: {e}")

if __name__ == "__main__":
    add_tag_column()