# EasyML Deployment Guide

## Production Deployment

### 1. Database Setup

#### PostgreSQL Setup
```sql
-- Create database and user
CREATE DATABASE easyml_db;
CREATE USER easyml_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE easyml_db TO easyml_user;
```

#### MongoDB Setup
```javascript
// Create database and user
use easyml
db.createUser({
  user: "easyml_user",
  pwd: "secure_password_here",
  roles: [
    { role: "readWrite", db: "easyml" },
    { role: "dbAdmin", db: "easyml" }
  ]
})
```

### 2. Environment Configuration

Create production `.env` file:
```env
# Database Configuration
POSTGRES_URL=postgresql://easyml_user:secure_password@your-postgres-host:5432/easyml_db
MONGO_URL=mongodb://easyml_user:secure_password@your-mongo-host:27017/easyml
MONGO_DB_NAME=easyml

# Security (CHANGE THESE!)
SECRET_KEY=your-super-secure-random-secret-key-256-bits
JWT_SECRET_KEY=your-jwt-secret-key-different-from-above

# DVC Configuration
DVC_REMOTE_URL=s3://your-production-bucket/easyml-data
DVC_REMOTE_NAME=production

# MLflow Configuration
MLFLOW_TRACKING_URI=postgresql://easyml_user:secure_password@your-postgres-host:5432/easyml_mlflow

# Application Settings
ENVIRONMENT=production
DEBUG=false
```

### 3. AWS S3 Setup for DVC

1. Create S3 bucket for DVC storage
2. Create IAM user with S3 access
3. Configure AWS credentials:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 4. Server Deployment

#### Using Docker (Recommended)

1. Build the image:
```bash
docker build -t easyml-api .
```

2. Run with environment:
```bash
docker run -d \
  --name easyml-api \
  --env-file .env \
  -p 8000:8000 \
  -v /app/uploads:/app/uploads \
  -v /app/dvc_storage:/app/dvc_storage \
  easyml-api
```

#### Using systemd service

1. Create service file `/etc/systemd/system/easyml.service`:
```ini
[Unit]
Description=EasyML API Service
After=network.target

[Service]
Type=simple
User=easyml
WorkingDirectory=/opt/easyml
ExecStart=/opt/easyml/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
EnvironmentFile=/opt/easyml/.env
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. Enable and start:
```bash
sudo systemctl enable easyml
sudo systemctl start easyml
```

### 5. Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 6. SSL/HTTPS Setup

```bash
# Using Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 7. Database Initialization

```bash
# Run database initialization
python scripts/init_database.py

# The script will create:
# - All PostgreSQL tables
# - MongoDB collections and indexes
# - Default admin user (admin/admin123)
# - DVC configuration
```

### 8. Health Checks

The API provides health check endpoints:
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system status
- `GET /dvc/status` - DVC configuration status

### 9. Monitoring

#### Application Logs
```bash
# View application logs
journalctl -u easyml -f

# Or with Docker
docker logs -f easyml-api
```

#### Database Monitoring
```sql
-- PostgreSQL connection monitoring
SELECT * FROM pg_stat_activity WHERE datname = 'easyml_db';

-- MongoDB connection monitoring
db.runCommand({serverStatus: 1})
```

### 10. Backup Strategy

#### PostgreSQL Backup
```bash
# Daily backup
pg_dump -h your-postgres-host -U easyml_user easyml_db > backup_$(date +%Y%m%d).sql
```

#### MongoDB Backup
```bash
# Daily backup
mongodump --host your-mongo-host --db easyml --out backup_$(date +%Y%m%d)
```

#### DVC/S3 Backup
S3 bucket versioning should be enabled for automatic versioning of DVC-tracked files.

### 11. Security Checklist

- [ ] Change default admin password after first login
- [ ] Use strong, unique passwords for all services
- [ ] Enable database SSL/TLS connections
- [ ] Configure firewall to restrict database access
- [ ] Set up API rate limiting
- [ ] Enable HTTPS with valid SSL certificates
- [ ] Regularly update dependencies
- [ ] Monitor security logs
- [ ] Set up database connection encryption
- [ ] Use environment variables for all secrets

### 12. Scaling Considerations

#### Horizontal Scaling
- Deploy multiple API instances behind a load balancer
- Use external session storage (Redis)
- Implement database connection pooling

#### Database Scaling
- PostgreSQL: Read replicas for read-heavy workloads
- MongoDB: Sharding for large datasets
- Connection pooling and optimization

#### Storage Scaling
- S3 for unlimited DVC storage
- CDN for serving static content
- Separate storage for different data types

### 13. Performance Optimization

1. Database indexing (already configured in init script)
2. Connection pooling
3. Caching frequent queries
4. Async processing for heavy ML tasks
5. Background job processing for model training

### 14. Disaster Recovery

1. Regular automated backups
2. Database replication
3. Multi-region S3 storage
4. Infrastructure as Code (IaC)
5. Documented recovery procedures

This deployment guide ensures a production-ready, scalable, and secure EasyML platform with proper data isolation and automated DVC integration.
