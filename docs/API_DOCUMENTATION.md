# EasyML API Documentation

## Overview
EasyML is a multi-tenant machine learning platform with automated DVC (Data Version Control) integration. Each user can have multiple projects, and all datasets and models are automatically versioned with proper isolation.

## Architecture
- **PostgreSQL**: Structured data (users, projects, experiments, model versions)
- **MongoDB**: Flexible schemas (sessions, DVC metadata, MLflow runs, audit logs)
- **DVC**: Automated data/model versioning with user/project isolation (Azure Blob Storage backend)
- **MLflow**: Experiment tracking
- **FastAPI**: REST API with JWT authentication

## Authentication

All endpoints (except registration and login) require authentication via JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

### Authentication Endpoints

#### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "full_name": "string"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User created successfully",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "user@example.com",
    "full_name": "string",
    "is_active": true,
    "is_superuser": false
  }
}
```

#### POST /auth/login
Login with username/email and password.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "jwt_token_string",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "user@example.com"
  }
}
```

## Project Management

### Project Endpoints

#### POST /projects
Create a new project.

**Request Body:**
```json
{
  "name": "My ML Project",
  "description": "Project description",
  "project_type": "classification"
}
```

#### GET /projects
List all projects for the authenticated user.

**Response:**
```json
{
  "success": true,
  "projects": [
    {
      "id": "uuid",
      "name": "string",
      "description": "string",
      "project_type": "classification",
      "created_at": "2025-07-24T10:00:00Z",
      "updated_at": "2025-07-24T10:00:00Z"
    }
  ],
  "total_count": 1
}
```

#### GET /projects/{project_id}
Get project details.

#### PUT /projects/{project_id}
Update project details.

#### DELETE /projects/{project_id}
Delete a project and all associated data.

## Data Version Control (DVC)

### DVC Endpoints

#### POST /dvc/projects/{project_id}/models/version
Upload and version a model file.

**Request:**
- `model_file`: Model file (multipart/form-data)
- `model_name`: Name of the model
- `metadata`: Optional JSON metadata

**Response:**
```json
{
  "success": true,
  "version_id": "uuid",
  "storage_path": "string",
  "dvc_path": "string",
  "version": "v_1721822400",
  "hash": "md5_hash",
  "size_bytes": 1024
}
```

#### POST /dvc/projects/{project_id}/datasets/version
Upload and version a dataset file.

**Request:**
- `dataset_file`: Dataset file (multipart/form-data)
- `dataset_name`: Name of the dataset
- `metadata`: Optional JSON metadata

#### GET /dvc/projects/{project_id}/models/versions
List all model versions for a project.

**Response:**
```json
{
  "success": true,
  "versions": [
    {
      "name": "model",
      "version": "v_1721822400",
      "hash": "md5_hash",
      "size_bytes": 1024,
      "created_at": "2025-07-24T10:00:00Z",
      "metadata": {}
    }
  ],
  "total_count": 1
}
```

#### GET /dvc/projects/{project_id}/datasets/versions
List all dataset versions for a project.

#### GET /dvc/projects/{project_id}/models/{name}/versions/{version}
Retrieve a specific model version.

#### DELETE /dvc/projects/{project_id}/cleanup?keep_latest=5
Clean up old versions, keeping only the latest N versions.

#### GET /dvc/status
Get DVC configuration status.

## Model Training

### Training Endpoints

#### POST /training/projects/{project_id}/train
Train a machine learning model with automatic DVC versioning.

**Request Body:**
```json
{
  "file_id": "string",
  "target_column": "string",
  "model_type": "random_forest",
  "test_size": 0.2,
  "hyperparameters": {},
  "use_cross_validation": true,
  "cv_folds": 5,
  "session_id": "string",
  "auto_version": true
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "uuid",
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.96,
  "f1_score": 0.95,
  "train_metrics": {},
  "test_metrics": {},
  "feature_importance": {},
  "confusion_matrix": [],
  "dvc_info": {
    "model_version": "v_1721822400",
    "storage_path": "string"
  }
}
```

#### GET /training/models
Get list of available model types.

#### GET /training/models/{model_type}/hyperparameters
Get default hyperparameters for a model type.

## File Upload

### Upload Endpoints

#### POST /upload/file
Upload a data file for processing.

**Request:**
- `file`: Data file (CSV, JSON, Excel)

**Response:**
```json
{
  "success": true,
  "file_id": "uuid",
  "filename": "data.csv",
  "size": 1024,
  "columns": ["col1", "col2", "target"],
  "rows": 1000
}
```

## Data Processing

### Preprocessing Endpoints

#### POST /preprocessing/analyze
Analyze uploaded data and get preprocessing suggestions.

#### POST /preprocessing/apply
Apply preprocessing operations to data.

### Visualization Endpoints

#### POST /visualization/create
Create data visualizations.

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-07-24T10:00:00Z"
}
```

Common HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Rate Limiting

API endpoints are rate-limited:
- Authentication endpoints: 5 requests per minute
- File upload endpoints: 10 requests per minute
- Other endpoints: 100 requests per minute

## Data Isolation

The platform ensures complete data isolation:
- Each user can only access their own projects
- DVC storage is organized by user_id/project_id
- Database queries are filtered by user permissions
- File uploads are scoped to user projects

## Example Workflow

1. **Register/Login**: Create account and get JWT token
2. **Create Project**: POST to `/projects`
3. **Upload Data**: POST to `/upload/file`
4. **Train Model**: POST to `/training/projects/{project_id}/train` with `auto_version: true`
5. **Model Automatically Versioned**: DVC handles versioning without manual intervention
6. **Access Versions**: GET `/dvc/projects/{project_id}/models/versions`

The system automatically:
- Versions your trained models with DVC
- Stores metadata in databases
- Tracks experiments with MLflow
- Maintains complete user/project isolation
- Provides audit trails

No manual DVC commands needed - everything is automated!
