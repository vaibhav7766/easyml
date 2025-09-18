"""
Add missing is_active column to projects table
"""
from app.core.database import engine
from sqlalchemy import text

def fix_projects_table():
    """Add is_active column to projects table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            # Check if is_active column exists in projects table
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'projects' 
                AND column_name = 'is_active'
            """))
            
            if not result.fetchone():
                # Add the is_active column
                conn.execute(text("""
                    ALTER TABLE projects 
                    ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE
                """))
                
                conn.commit()
                print('✅ Successfully added is_active column to projects table')
            else:
                print('ℹ️ is_active column already exists in projects table')
                
    except Exception as e:
        print(f'❌ Error adding is_active column: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_projects_table()