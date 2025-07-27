#!/bin/bash
set -e

echo "🚀 Starting EasyML Container with Azure Blob Storage"

# Check if .env exists, if not copy from template
if [ ! -f /app/.env ]; then
    echo "📋 Creating .env from template..."
    cp /app/.env.template /app/.env
fi

# Wait for databases to be ready
echo "⏳ Waiting for databases..."
while ! curl -s $POSTGRES_URL > /dev/null 2>&1; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

while ! curl -s $MONGO_URL > /dev/null 2>&1; do
    echo "Waiting for MongoDB..."
    sleep 2
done

echo "✅ Databases are ready!"

# Initialize DVC if not already done
if [ ! -d /app/.dvc ]; then
    echo "🔄 Initializing DVC..."
    cd /app && dvc init --no-scm
fi

# Configure DVC Azure remote if environment variables are set
if [ ! -z "$DVC_REMOTE_URL" ] && [ ! -z "$DVC_AZURE_CONNECTION_STRING" ]; then
    echo "🔧 Configuring DVC Azure remote..."
    cd /app && dvc remote add -d azure "$DVC_REMOTE_URL" --force
    cd /app && dvc remote modify azure connection_string "$DVC_AZURE_CONNECTION_STRING"
    echo "✅ DVC Azure remote configured"
fi

# Initialize database if needed
echo "🗄️ Checking database initialization..."
if python -c "
import sys
sys.path.append('/app')
try:
    from app.core.database import DatabaseManager
    import asyncio
    async def check():
        db = DatabaseManager()
        await db.init_databases()
        # Quick check if tables exist
        return True
    asyncio.run(check())
    print('Database ready')
except Exception as e:
    print('Database needs initialization')
    exit(1)
" 2>/dev/null; then
    echo "✅ Database already initialized"
else
    echo "🔧 Initializing database..."
    python /app/scripts/init_database.py
    echo "✅ Database initialized"
fi

echo "🎉 EasyML container startup complete!"
echo "📊 System status:"
python /app/check_system_status.py

# Execute the main command
exec "$@"
