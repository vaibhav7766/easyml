# EasyML - Complete ML Platform Architecture

## 🏗️ Architecture Overview

EasyML now features a comprehensive multi-database architecture integrating:

- **PostgreSQL** - Structured data (users, projects, experiments, model versions)
- **MongoDB** - Document storage (sessions, DVC metadata, audit logs)
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data and model versioning
- **Redis** - Session management and caching

## 🗂️ Project Structure

```
easyml/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── projects.py      # Project management
│   │   ├── training.py      # Enhanced ML training
│   │   └── ...
│   ├── core/
│   │   ├── auth.py          # Authentication service
│   │   ├── database.py      # Multi-database config
│   │   └── config.py        # App configuration
│   ├── models/
│   │   ├── sql_models.py    # PostgreSQL models
│   │   └── mongo_schemas.py # MongoDB schemas
│   └── services/
│       ├── project_service.py           # Project management
│       ├── enhanced_model_training.py   # Enhanced ML training
│       └── ...
├── models/           # Model storage (organized by project)
├── datasets/         # Dataset storage (organized by project)
├── mlruns/          # MLflow artifacts
├── scripts/
│   ├── init_database.py     # Database initialization
│   ├── postgres_init.sql    # PostgreSQL setup
│   └── mongo_init.js        # MongoDB setup
├── docker-compose.yml       # Database services
├── setup.sh                 # Complete setup script
└── requirements.txt         # Updated dependencies
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd easyml

# Run the automated setup
./setup.sh
```

### 2. Manual Setup (Alternative)

```bash
# 1. Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# 2. Start databases
docker-compose up -d

# 3. Install dependencies  
pip install -r requirements.txt

# 4. Initialize databases
python scripts/init_database.py

# 5. Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🗄️ Database Architecture

### PostgreSQL Tables

- **users** - User authentication and profiles
- **projects** - Project organization
- **ml_experiments** - Experiment tracking (linked to MLflow)
- **model_versions** - Model versioning and metadata
- **dataset_versions** - Dataset versioning
- **model_deployments** - Deployment tracking

### MongoDB Collections

- **model_sessions** - Active ML training sessions
- **dvc_metadata** - DVC file tracking and metadata
- **mlflow_runs** - MLflow run metadata
- **project_configs** - Project configurations
- **audit_logs** - Complete audit trail
- **training_errors** - Error tracking

## 🔐 Authentication System

### Endpoints

```bash
# Register new user
POST /api/v1/auth/register
{
  "username": "johndoe",
  "email": "john@example.com", 
  "password": "secure123",
  "full_name": "John Doe"
}

# Login
POST /api/v1/auth/login
{
  "username": "johndoe",
  "password": "secure123"
}

# Get current user
GET /api/v1/auth/me
Authorization: Bearer <token>
```

### Default Credentials

```
Username: admin
Password: admin123
⚠️ Change these in production!
```

## 📁 Project Management

### Creating Projects

```bash
# Create new project
POST /api/v1/projects/
Authorization: Bearer <token>
{
  "name": "My ML Project",
  "description": "Classification project",
  "mlflow_experiment_name": "classification_exp"
}
```

### Project Structure

Each project gets its own:
- Model storage folder: `models/{project_id}/`
- Dataset storage folder: `datasets/{project_id}/`
- MLflow experiment
- MongoDB configuration document

## 🤖 Enhanced ML Training

### Features

- **Project Integration** - All training linked to projects
- **Database Persistence** - Sessions and experiments stored
- **MLflow Integration** - Automatic experiment tracking
- **DVC Versioning** - Model and data versioning
- **Audit Trail** - Complete action logging

### Usage

```bash
# Train model with project context
POST /api/v1/training/train
Authorization: Bearer <token>
{
  "file_id": "data.csv",
  "target_column": "target",
  "model_type": "random_forest_classifier",
  "project_id": "project-uuid",
  "session_id": "session-123"
}
```

## 📊 Model Storage & Versioning

### Storage Hierarchy

```
models/
├── {project_id}/
│   ├── model_{uuid}/
│   │   ├── model.joblib
│   │   ├── metadata.json
│   │   └── model.joblib.dvc
│   └── ...
└── ...
```

### DVC Integration

- Automatic file tracking
- Version metadata in MongoDB
- Support for remote storage (S3, GCS, etc.)

## 🔬 MLflow Integration

### Features

- **Experiment Tracking** - Parameters, metrics, artifacts
- **Model Registry** - Centralized model management
- **Database Backend** - PostgreSQL for metadata
- **Project Organization** - Experiments per project

### Access

- MLflow UI: http://localhost:5000
- Automatic experiment creation per project
- Model artifacts stored in project folders

## 🐳 Services

### Docker Services

```bash
# Start all services
docker-compose up -d

# Stop all services  
docker-compose down

# View logs
docker-compose logs -f

# Reset everything
docker-compose down -v
```

### Running Services

- **PostgreSQL**: localhost:5432
- **MongoDB**: localhost:27017  
- **MLflow**: http://localhost:5000
- **Redis**: localhost:6379
- **EasyML API**: http://localhost:8000

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run training endpoint tests
pytest tests/test_training_endpoints.py -v

# Test specific functionality
pytest tests/test_auth.py -v
pytest tests/test_projects.py -v
```

### Test Database Integration

The test suite now includes:
- Multi-database integration tests
- Authentication flow tests
- Project management tests  
- Enhanced training pipeline tests

## 📖 API Documentation

### Swagger UI
http://localhost:8000/docs

### New Endpoint Categories

1. **Authentication** (`/api/v1/auth/`)
   - User registration and login
   - Token management
   - Profile management

2. **Projects** (`/api/v1/projects/`)
   - Project CRUD operations
   - Project configuration
   - Storage path management

3. **Enhanced Training** (`/api/v1/training/`)
   - Project-aware training
   - Database-persistent sessions
   - Integrated versioning

## 🔧 Configuration

### Environment Variables

```bash
# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# PostgreSQL
POSTGRES_URL=postgresql://user:pass@localhost:5432/easyml_db

# MongoDB  
MONGO_URI=mongodb://localhost:27017

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@localhost:5432/easyml_mlflow

# Storage
MODELS_DIR=./models
DATASETS_DIR=./datasets
```

## 🚀 Production Deployment

### Security Checklist

- [ ] Change default admin credentials
- [ ] Generate secure SECRET_KEY
- [ ] Configure proper database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up backup strategies

### Scalability Considerations

- Use external PostgreSQL/MongoDB services
- Configure Redis cluster for session management
- Set up MLflow with S3 backend
- Use load balancers for API instances

## 🔍 Monitoring & Logging

### Audit Trail

All actions are logged to MongoDB:
- User authentication
- Project operations
- Model training sessions
- File operations

### Error Tracking

- Training errors logged to MongoDB
- Comprehensive error context
- Session-based error correlation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `pytest tests/ -v`
4. Submit pull request

## 📝 License

MIT License - see LICENSE file for details.

---

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check services
   docker-compose ps
   
   # Restart services
   docker-compose restart postgres mongodb
   ```

2. **Permission Errors**
   ```bash
   # Fix folder permissions
   chmod -R 755 models datasets uploads
   ```

3. **MLflow Issues**
   ```bash
   # Restart MLflow
   docker-compose restart mlflow
   ```

### Support

- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- MLflow UI: http://localhost:5000
