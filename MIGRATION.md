# EasyML Migration Guide

## Overview

The EasyML project has been refactored from a monolithic structure to a modular FastAPI architecture following best practices. This guide explains the changes and how to migrate.

## 🏗️ **Structural Changes**

### Before (Monolithic)
```
easyml/
├── main.py              # All endpoints in one file
├── visualization.py     # Visualization logic
├── preprocessing.py     # Preprocessing logic
├── models.py           # Database models
├── enums.py            # Enumerations
├── db.py               # Database connection
├── crud.py             # CRUD operations
└── metrics.py          # Metrics logic
```

### After (Modular)
```
easyml/
├── app/
│   ├── main.py              # New main FastAPI app
│   ├── api/
│   │   └── v1/
│   │       ├── api.py           # API router
│   │       └── endpoints/
│   │           ├── upload.py        # File upload endpoints
│   │           ├── visualization.py # Visualization endpoints
│   │           ├── preprocessing.py # Preprocessing endpoints
│   │           └── training.py      # Training endpoints
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── database.py      # Database connection
│   │   └── enums.py         # Enhanced enumerations
│   ├── models/
│   │   └── database.py      # Database models and CRUD
│   ├── schemas/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── visualization.py     # Visualization service
│   │   ├── preprocessing.py     # Preprocessing service
│   │   ├── model_training.py    # ML training service
│   │   └── file_service.py      # File management service
│   └── utils/
├── main.py              # Legacy compatibility file
├── requirements.txt     # Updated dependencies
└── README.md           # Comprehensive documentation
```

## 🔄 **API Changes**

### Old Endpoints vs New Endpoints

| Old Endpoint | New Endpoint | Changes |
|-------------|-------------|---------|
| `POST /upload` | `POST /api/v1/upload/` | ✅ Enhanced with file validation |
| `POST /preprocessing` | `POST /api/v1/preprocessing/apply` | ✅ Improved with recommendations |
| `GET /visualizations` | `POST /api/v1/visualization/generate` | ✅ More plot types, batch support |
| `GET /dashboard` | ❌ Removed | Use individual endpoints |
| `POST /feature_selection` | ✅ Integrated into preprocessing | |
| `POST /choosing_models` | `POST /api/v1/training/train` | ✅ Enhanced with more models |
| `GET /results` | `GET /api/v1/training/model-info/{session_id}` | ✅ Better session management |

## 📦 **New Features Added**

### 1. Enhanced Visualization (18+ Plot Types)
- All "In Progress" plots from Notion database implemented
- Batch visualization generation
- Automatic plot recommendations based on data types
- Better error handling and validation

### 2. Improved Preprocessing
- Automatic preprocessing recommendations
- Multiple imputation methods (mean, median, mode, KNN, forward/backward fill)
- Advanced encoding (one-hot, label, ordinal)
- Multiple normalization techniques (standard, min-max, robust, max-abs)
- Data cleaning with outlier detection

### 3. Comprehensive Model Training
- Support for 15+ ML algorithms
- Hyperparameter tuning with GridSearchCV
- Cross-validation support
- Model persistence and export
- Session-based model management
- Feature importance analysis

### 4. File Management Service
- Support for CSV, Excel, JSON, Parquet files
- File validation and size limits
- Data preview without full loading
- Export processed data in multiple formats

## 🚀 **Migration Steps**

### 1. Update Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 3. Create Upload Directory
```bash
mkdir uploads
```

### 4. Update Import Statements
If you have external code using the old structure:

**Before:**
```python
from visualization import Visualization
from preprocessing import Preprocessing
from enums import Plots, Options
```

**After:**
```python
from app.services.visualization import VisualizationService
from app.services.preprocessing import PreprocessingService
from app.core.enums import PlotType, PreprocessingOption
```

### 5. Update API Calls
**Before:**
```python
# Old API call
response = requests.get("http://localhost:8000/visualizations", params={
    "project_id": "123",
    "user_id": "user1", 
    "plot_type": "histogram",
    "x": "column1"
})
```

**After:**
```python
# New API call
response = requests.post("http://localhost:8000/api/v1/visualization/generate", json={
    "file_id": "uploaded_file_id",
    "plot_type": "histogram",
    "x_column": "column1"
})
```

## 🔧 **Configuration Changes**

### Database Configuration
The new system uses environment variables for configuration:

**Before (hardcoded):**
```python
client = MongoClient("mongodb://localhost:27017")
```

**After (configurable):**
```python
from app.core.config import get_settings
settings = get_settings()
client = MongoClient(settings.mongo_uri)
```

### File Upload Configuration
**Before:**
```python
# Hardcoded upload directory
os.makedirs("uploads", exist_ok=True)
```

**After:**
```python
# Configurable upload directory
from app.core.config import get_settings
settings = get_settings()
upload_dir = Path(settings.upload_dir)
```

## 🧪 **Testing the Migration**

### 1. Start the Application
```bash
# Using new structure (recommended)
python -m app.main

# Or using legacy compatibility
uvicorn main:app --reload

# Or direct uvicorn
uvicorn app.main:app --reload
```

### 2. Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# API information
curl http://localhost:8000/

# Upload a file
curl -X POST "http://localhost:8000/api/v1/upload/" \
     -F "file=@test_data.csv"

# Generate visualization
curl -X POST "http://localhost:8000/api/v1/visualization/generate" \
     -H "Content-Type: application/json" \
     -d '{"file_id": "your-file-id", "plot_type": "histogram", "x_column": "column_name"}'
```

### 3. Check Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 **Benefits of New Architecture**

### 1. **Scalability**
- Modular services can be scaled independently
- Clear separation of concerns
- Easy to add new features

### 2. **Maintainability**
- Smaller, focused modules
- Better code organization
- Easier testing and debugging

### 3. **Extensibility**
- Plugin-like architecture
- Easy to add new ML algorithms
- Configurable components

### 4. **Production Ready**
- Environment-based configuration
- Proper error handling
- Comprehensive logging
- API versioning

## 🐛 **Common Issues & Solutions**

### Issue 1: Import Errors
**Problem:** `ModuleNotFoundError: No module named 'app'`

**Solution:** Make sure you're running from the project root directory and have installed dependencies.

### Issue 2: File Upload Errors
**Problem:** File upload fails with permission errors

**Solution:** Check upload directory permissions and ensure it exists:
```bash
mkdir uploads
chmod 755 uploads
```

### Issue 3: Database Connection Errors
**Problem:** MongoDB connection fails

**Solution:** Update your `.env` file with correct MongoDB URI:
```env
MONGO_URI=mongodb://localhost:27017
```

### Issue 4: Legacy Code Compatibility
**Problem:** Old API calls return 404

**Solution:** Update API endpoints to use `/api/v1/` prefix and new request format.

## 📚 **Additional Resources**

- **API Documentation:** http://localhost:8000/docs
- **Code Examples:** See README.md
- **Configuration:** See app/core/config.py
- **Service Documentation:** See individual service files

## 🤝 **Getting Help**

If you encounter issues during migration:

1. Check the logs for detailed error messages
2. Verify your environment configuration
3. Test individual components using the API documentation
4. Create an issue with detailed error information

---

**Migration Complete!** 🎉

The new modular architecture provides better scalability, maintainability, and extensibility while maintaining backward compatibility through the legacy main.py file.
