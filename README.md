# EasyML - No-Code Machine Learning Platform

A comprehensive, modular FastAPI-based platform for no-code machine learning operations including data visualization, preprocessing, model training, and predictions.

## 🚀 Features

### 📊 **Data Visualization**
- **18+ Plot Types**: Histogram, Scatter, Line, Bar, Box, Violin, Heatmap, Pair Plot, PCA, and more
- **Interactive Visualizations**: Correlation matrices, feature importance, residual plots
- **Batch Processing**: Generate multiple visualizations simultaneously
- **Automatic Recommendations**: Smart plot suggestions based on data types

### 🔧 **Data Preprocessing**
- **Automated Data Cleaning**: Missing value imputation, outlier detection
- **Feature Engineering**: Encoding, normalization, scaling
- **Smart Recommendations**: AI-powered preprocessing suggestions
- **Export Processed Data**: Multiple format support (CSV, Excel, JSON, Parquet)

### 🤖 **Machine Learning**
- **Multiple Algorithms**: Linear/Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Evaluation**: Comprehensive metrics for classification and regression
- **Model Persistence**: Save/load trained models

### 📁 **File Management**
- **Multi-format Support**: CSV, Excel, JSON, Parquet files
- **File Validation**: Size limits, format checking
- **Data Preview**: Quick data exploration without full loading
- **Batch Operations**: Process multiple files efficiently

## 🏗️ **Architecture**

```
easyml/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── api.py              # Main API router
│   │       └── endpoints/
│   │           ├── upload.py       # File upload endpoints
│   │           ├── visualization.py # Visualization endpoints
│   │           ├── preprocessing.py # Data preprocessing endpoints
│   │           └── training.py     # Model training endpoints
│   ├── core/
│   │   ├── config.py              # Application configuration
│   │   ├── database.py            # Database connectivity
│   │   └── enums.py               # Application enumerations
│   ├── models/
│   │   └── database.py            # Database models and CRUD
│   ├── schemas/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── visualization.py       # Visualization service logic
│   │   ├── preprocessing.py       # Data preprocessing service
│   │   ├── model_training.py      # ML training service
│   │   └── file_service.py        # File management service
│   ├── utils/
│   │   └── __init__.py
│   └── main.py                    # FastAPI application entry point
├── tests/                         # Test files
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 **Installation & Setup**

### Prerequisites
- Python 3.8+
- MongoDB (optional, for data persistence)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd easyml
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# Application Settings
DEBUG=true
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# File Upload Settings
UPLOAD_DIR=./uploads
MAX_PLOT_FEATURES=20

# Database Settings (optional)
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=easyml
```

### 5. Create Upload Directory
```bash
mkdir uploads
```

### 6. Run the Application
```bash
# Development mode
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 **API Documentation**

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### 🔄 **Health & Info**
- `GET /` - API information
- `GET /health` - Health check

#### 📁 **File Management** (`/api/v1/upload/`)
- `POST /` - Upload data file
- `GET /preview/{file_id}` - Preview data
- `GET /list` - List uploaded files
- `GET /info/{file_id}` - Get file information
- `DELETE /{file_id}` - Delete file

#### 📊 **Visualization** (`/api/v1/visualization/`)
- `POST /generate` - Generate single visualization
- `POST /batch` - Generate multiple visualizations
- `GET /plot-types` - Available plot types
- `GET /data-info/{file_id}` - Data info for plotting

#### 🔧 **Preprocessing** (`/api/v1/preprocessing/`)
- `POST /apply` - Apply preprocessing operations
- `POST /delete-columns` - Delete columns
- `GET /data-summary/{file_id}` - Data summary
- `GET /recommendations/{file_id}` - Preprocessing recommendations
- `POST /export` - Export processed data

#### 🤖 **Model Training** (`/api/v1/training/`)
- `POST /train` - Train ML model
- `POST /hyperparameter-tuning` - Tune hyperparameters
- `POST /predict` - Make predictions
- `GET /model-info/{session_id}` - Model information
- `GET /training-history/{session_id}` - Training history
- `GET /available-models` - Available model types
- `POST /save-model/{session_id}` - Save model
- `GET /export-model/{session_id}` - Export model

## 🎯 **Usage Examples**

### 1. Upload and Visualize Data
```python
import requests

# Upload file
files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/api/v1/upload/', files=files)
file_id = response.json()['file_id']

# Generate visualization
viz_request = {
    "file_id": file_id,
    "plot_type": "histogram",
    "x_column": "age",
    "plot_params": {"bins": 30}
}
response = requests.post('http://localhost:8000/api/v1/visualization/generate', json=viz_request)
plot_base64 = response.json()['plot_base64']
```

### 2. Preprocess Data
```python
# Apply preprocessing
preprocess_request = {
    "file_id": file_id,
    "operations": {
        "imputation": "age",
        "encoding": "category",
        "normalization": "income"
    }
}
response = requests.post('http://localhost:8000/api/v1/preprocessing/apply', json=preprocess_request)
```

### 3. Train Model
```python
# Train ML model
training_request = {
    "file_id": file_id,
    "target_column": "target",
    "model_type": "random_forest_classifier",
    "test_size": 0.2,
    "session_id": "my_session"
}
response = requests.post('http://localhost:8000/api/v1/training/train', json=training_request)
```

## 🧪 **Testing**

Run tests using pytest:
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 🔧 **Configuration**

### Environment Variables
- `DEBUG`: Enable debug mode (default: False)
- `CORS_ORIGINS`: Allowed CORS origins (JSON list)
- `UPLOAD_DIR`: File upload directory (default: ./uploads)
- `MAX_PLOT_FEATURES`: Max features for plotting (default: 20)
- `MONGO_URI`: MongoDB connection string
- `DATABASE_NAME`: Database name for persistence

### Application Settings
Modify `app/core/config.py` for additional configuration options.

## 📦 **Deployment**

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use production WSGI server (Gunicorn)
- Set up proper logging
- Configure reverse proxy (Nginx)
- Implement authentication/authorization
- Set up monitoring and metrics
- Use external file storage (S3, etc.)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- Create an issue for bug reports or feature requests
- Check the documentation at `/docs` for API details
- Review the code examples in this README

## 🚀 **Roadmap**

- [ ] Real-time data streaming
- [ ] Advanced AutoML capabilities
- [ ] Model deployment endpoints
- [ ] WebSocket support for long-running operations
- [ ] Advanced visualization dashboard
- [ ] Model interpretability features
- [ ] A/B testing framework
- [ ] Data versioning and lineage tracking

---

**EasyML** - Making machine learning accessible to everyone! 🎯
