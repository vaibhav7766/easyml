# 📁 EasyML Project Structure

## Core Application
```
app/
├── __init__.py                 # App package initialization
├── main.py                     # FastAPI application entry point
├── api/                        # API layer
│   ├── __init__.py
│   └── v1/                     # API version 1
│       ├── __init__.py
│       ├── api.py              # API router setup
│       └── endpoints/          # API endpoints
├── core/                       # Core application logic
│   ├── __init__.py
│   ├── auth.py                 # Authentication logic
│   ├── config.py               # Configuration management
│   ├── database.py             # Database connections
│   └── enums.py                # Application enumerations
├── models/                     # Data models
│   ├── __init__.py
│   ├── database.py             # Database model base
│   ├── mongo_schemas.py        # MongoDB schemas
│   └── sql_models.py           # PostgreSQL models
├── schemas/                    # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py              # API request/response schemas
├── services/                   # Business logic services
│   ├── __init__.py
│   ├── dvc_service.py          # DVC version control
│   ├── enhanced_model_training.py  # ML training service
│   ├── file_service.py         # File management
│   ├── model_training.py       # Model training utilities
│   ├── preprocessing.py        # Data preprocessing
│   ├── project_service.py      # Project management
│   └── visualization.py        # Data visualization
└── utils/                      # Utility functions
    └── __init__.py
```

## Configuration & Scripts
```
scripts/                        # Database initialization scripts
├── init_database.py           # PostgreSQL initialization
├── mongo_init.js              # MongoDB initialization
└── postgres_init.sql          # PostgreSQL schema

docs/                           # Documentation
├── API_DOCUMENTATION.md       # API documentation
├── AZURE_DVC_SETUP.md         # Azure DVC setup guide
├── DEPLOYMENT_GUIDE.md        # Deployment instructions
└── TRAINING_ENDPOINTS_DOCUMENTATION.md  # Training API docs
```

## Data & Storage
```
datasets/                       # Dataset storage (DVC managed)
├── .gitkeep

uploads/                        # Temporary file uploads
├── .gitkeep

models/                         # Trained model storage (DVC managed)
├── .gitkeep

logs/                          # Application logs
├── .gitkeep

mlruns/                        # MLflow experiment tracking
```

## Testing
```
tests/                         # Test suite
├── __init__.py
├── test_training_endpoints.py # Training API tests
├── test_visualizations.py     # Visualization tests
└── test_workflow.py           # End-to-end workflow tests
```

## Container & Deployment
```
Dockerfile                     # Multi-stage container build
docker-compose.yml            # Docker Compose setup
docker-entrypoint.sh          # Container startup script
test_container.sh             # Container testing script
```

## Configuration Files
```
.env                          # Environment variables
.env.example                  # Environment template
.env.container               # Container-specific config
.envrc                       # Direnv configuration
.dvcignore                   # DVC ignore patterns
.gitignore                   # Git ignore patterns
requirements.txt             # Python dependencies
uploads.dvc                  # DVC tracking file
check_system_status.py       # System health checker
```

## Key Features

### 🔐 **Authentication & Multi-tenancy**
- JWT-based authentication
- User and project isolation
- Role-based access control

### 📊 **Data Management**
- Automated DVC integration with Azure Blob Storage
- Multi-format dataset support (CSV, JSON)
- Version-controlled datasets and models

### 🤖 **Machine Learning**
- Multiple model types (classification, regression)
- Automated preprocessing pipelines
- MLflow experiment tracking
- Model versioning and deployment

### 🐳 **Container Deployment**
- Production-ready Docker containers
- Azure Blob Storage integration
- Health checks and monitoring
- Scalable deployment options

### 📈 **Visualization**
- 20+ chart types for data exploration
- Interactive dashboards
- Model performance visualization
- Data quality assessment

## Clean Structure Benefits

1. **Organized Codebase**: Clear separation of concerns
2. **Version Control**: DVC integration for data/model versioning
3. **Container Ready**: Production deployment with Docker
4. **Multi-tenant**: Complete user and project isolation
5. **Cloud Native**: Azure Blob Storage integration
6. **Testing**: Comprehensive test suite
7. **Documentation**: Complete API and setup documentation
