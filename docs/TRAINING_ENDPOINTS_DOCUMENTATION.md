# EasyML Training Endpoints Documentation

## Overview
This document provides comprehensive documentation for all training endpoints in the EasyML system. The API provides endpoints for machine learning model training, prediction, hyperparameter tuning, model management, and session handling.

## Base URL
```
http://localhost:8000/api/v1/training
```

## Authentication
Currently, no authentication is required for these endpoints.

## Available Endpoints

### 1. Get Available Models
**GET** `/available-models`

Returns a list of all available machine learning models with detailed information.

**Response:**
```json
{
  "success": true,
  "models": {
    "linear_regression": {
      "description": "Linear relationship modeling",
      "type": "regression",
      "hyperparameters": ["fit_intercept", "normalize"],
      "pros": ["Simple", "Interpretable", "Fast"],
      "cons": ["Assumes linearity", "Sensitive to outliers"]
    },
    // ... more models
  },
  "regression_models": [
    "linear_regression", "ridge", "lasso", "random_forest_regressor",
    "decision_tree_regressor", "svr", "knn_regressor", "elastic_net",
    "mlp_regressor", "xgboost_regressor", "lightgbm_regressor"
  ],
  "classification_models": [
    "logistic_regression", "random_forest_classifier", "decision_tree_classifier",
    "svc", "knn_classifier", "naive_bayes", "gradient_boosting_classifier",
    "mlp_classifier", "xgboost_classifier", "lightgbm_classifier"
  ]
}
```

**Available Models:**

#### Classification Models (10):
1. **Logistic Regression** - Linear model for classification
2. **Random Forest Classifier** - Ensemble of decision trees
3. **Decision Tree Classifier** - Single decision tree
4. **Support Vector Classifier (SVC)** - Support Vector Machine
5. **K-Nearest Neighbors Classifier** - Distance-based classification
6. **Naive Bayes** - Probabilistic classifier
7. **Gradient Boosting Classifier** - Sequential ensemble
8. **Multi-layer Perceptron Classifier** - Neural network
9. **XGBoost Classifier** - Extreme Gradient Boosting
10. **LightGBM Classifier** - Light Gradient Boosting Machine

#### Regression Models (11):
1. **Linear Regression** - Basic linear relationship modeling
2. **Ridge Regression** - Linear regression with L2 regularization
3. **Lasso Regression** - Linear regression with L1 regularization
4. **Random Forest Regressor** - Ensemble of decision trees
5. **Decision Tree Regressor** - Single decision tree
6. **Support Vector Regressor (SVR)** - Support Vector Machine
7. **K-Nearest Neighbors Regressor** - Distance-based regression
8. **Elastic Net** - Linear regression with L1 and L2 regularization
9. **Multi-layer Perceptron Regressor** - Neural network
10. **XGBoost Regressor** - Extreme Gradient Boosting
11. **LightGBM Regressor** - Light Gradient Boosting Machine

### 2. Get Model Hyperparameters
**GET** `/model-hyperparameters/{model_type}`

Returns default hyperparameters for a specific model type.

**Parameters:**
- `model_type` (path): The type of model (e.g., "random_forest_classifier")

**Response:**
```json
{
  "success": true,
  "model_type": "random_forest_classifier",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
  }
}
```

### 3. Train Model
**POST** `/train`

Trains a machine learning model on uploaded data.

**Request Body:**
```json
{
  "file_id": "uploaded_file_id.csv",
  "target_column": "target",
  "model_type": "random_forest_classifier",
  "session_id": "my_session_123",
  "test_size": 0.2,
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 15
  },
  "use_cross_validation": true,
  "cv_folds": 5
}
```

**Parameters:**
- `file_id` (required): ID of the uploaded data file
- `target_column` (required): Name of the target/label column
- `model_type` (required): Type of ML model to train
- `session_id` (optional): Session identifier for model management
- `test_size` (optional): Proportion of data for testing (0.1-0.5), default: 0.2
- `hyperparameters` (optional): Custom model hyperparameters
- `use_cross_validation` (optional): Whether to use cross-validation, default: true
- `cv_folds` (optional): Number of cross-validation folds, default: 5

**Response:**
```json
{
  "success": true,
  "model_type": "random_forest_classifier",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": 42
  },
  "is_classifier": true,
  "n_samples": 1000,
  "n_features": 10,
  "test_size": 0.2,
  "train_accuracy": 0.95,
  "test_accuracy": 0.92,
  "train_precision": 0.94,
  "test_precision": 0.91,
  "train_recall": 0.96,
  "test_recall": 0.93,
  "train_f1": 0.95,
  "test_f1": 0.92,
  "cv_scores": [0.91, 0.93, 0.89, 0.92, 0.94],
  "cv_mean": 0.918,
  "cv_std": 0.018,
  "cv_scoring": "accuracy",
  "session_id": "my_session_123"
}
```

### 4. Make Predictions
**POST** `/predict`

Makes predictions using a trained model.

**Request Body:**
```json
{
  "file_id": "prediction_data.csv",
  "session_id": "my_session_123"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [0, 1, 1, 0, 1],
  "model_type": "random_forest_classifier",
  "is_classifier": true,
  "probabilities": [
    [0.8, 0.2],
    [0.3, 0.7],
    [0.1, 0.9],
    [0.9, 0.1],
    [0.2, 0.8]
  ],
  "classes": ["0", "1"],
  "session_id": "my_session_123"
}
```

### 5. Hyperparameter Tuning
**POST** `/hyperparameter-tuning`

Performs automated hyperparameter optimization.

**Request Body:**
```json
{
  "file_id": "training_data.csv",
  "target_column": "target",
  "model_type": "random_forest_classifier",
  "session_id": "tuning_session",
  "param_grid": {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
  },
  "cv_folds": 5,
  "scoring": "accuracy"
}
```

**Response:**
```json
{
  "success": true,
  "best_params": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 2
  },
  "best_score": 0.924,
  "cv_results": {
    "mean_test_score": [0.89, 0.91, 0.924, 0.918],
    "params": [
      {"n_estimators": 50, "max_depth": 5},
      {"n_estimators": 100, "max_depth": 10},
      {"n_estimators": 200, "max_depth": 15},
      {"n_estimators": 200, "max_depth": 10}
    ]
  },
  "session_id": "tuning_session"
}
```

### 6. Get Model Information
**GET** `/model-info/{session_id}`

Returns information about a trained model in a session.

**Response:**
```json
{
  "success": true,
  "model_type": "random_forest_classifier",
  "is_classifier": true,
  "parameters": {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": 42
  },
  "has_feature_importance": true,
  "has_predict_proba": true,
  "session_id": "my_session_123"
}
```

### 7. Get Training History
**GET** `/training-history/{session_id}`

Returns the training history for a session.

**Response:**
```json
{
  "success": true,
  "training_history": [
    {
      "model_type": "random_forest_classifier",
      "test_accuracy": 0.92,
      "test_f1": 0.91,
      "cv_mean": 0.918,
      "timestamp": "2025-07-24T14:30:00Z"
    }
  ],
  "total_trainings": 1,
  "session_id": "my_session_123"
}
```

### 8. List Training Sessions
**GET** `/sessions`

Returns all active training sessions.

**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "session_1",
      "model_info": {
        "model_type": "random_forest_classifier",
        "is_classifier": true
      },
      "created_at": "2025-07-24T14:00:00Z"
    },
    {
      "session_id": "session_2",
      "model_info": {
        "model_type": "linear_regression",
        "is_classifier": false
      },
      "created_at": "2025-07-24T14:15:00Z"
    }
  ],
  "total_sessions": 2
}
```

### 9. Save Model
**POST** `/save-model/{session_id}`

Saves a trained model to file.

**Parameters:**
- `filename` (optional): Custom filename for the saved model

**Response:**
```json
{
  "success": true,
  "filepath": "/tmp/model_random_forest_classifier_my_session_123.joblib",
  "message": "Model saved successfully",
  "session_id": "my_session_123"
}
```

### 10. Export Model
**GET** `/export-model/{session_id}`

Exports a trained model as base64 encoded data.

**Response:**
```json
{
  "success": true,
  "model_data": "UEsDBBQAAAAIAFJV...", // base64 encoded model
  "model_type": "random_forest_classifier",
  "is_classifier": true,
  "size_bytes": 1024768,
  "session_id": "my_session_123"
}
```

### 11. Delete Session
**DELETE** `/session/{session_id}`

Deletes a training session and its associated model.

**Response:**
```json
{
  "success": true,
  "message": "Training session deleted successfully",
  "session_id": "my_session_123"
}
```

## Error Responses

### Common Error Codes:
- **400 Bad Request**: Invalid request parameters or data
- **404 Not Found**: Session or resource not found
- **500 Internal Server Error**: Server processing error

### Error Response Format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Usage Examples

### Complete Workflow Example:

1. **Get Available Models:**
```bash
curl -X GET "http://localhost:8000/api/v1/training/available-models"
```

2. **Train a Model:**
```bash
curl -X POST "http://localhost:8000/api/v1/training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "data.csv",
    "target_column": "target",
    "model_type": "random_forest_classifier",
    "session_id": "demo_session",
    "test_size": 0.2
  }'
```

3. **Make Predictions:**
```bash
curl -X POST "http://localhost:8000/api/v1/training/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "new_data.csv",
    "session_id": "demo_session"
  }'
```

4. **Get Model Info:**
```bash
curl -X GET "http://localhost:8000/api/v1/training/model-info/demo_session"
```

5. **Save Model:**
```bash
curl -X POST "http://localhost:8000/api/v1/training/save-model/demo_session"
```

## Best Practices

1. **Data Preparation**: Ensure your data is clean and properly formatted before training
2. **Feature Engineering**: Use appropriate preprocessing for categorical variables
3. **Model Selection**: Choose the right model type based on your problem (classification vs regression)
4. **Hyperparameter Tuning**: Use the hyperparameter tuning endpoint for optimal performance
5. **Cross-Validation**: Enable cross-validation for more robust model evaluation
6. **Session Management**: Use meaningful session IDs for better organization

## Notes

- All endpoints support CORS for web applications
- File uploads must be done through the file upload endpoints before training
- Models are stored in memory per session and will be lost when the server restarts
- For production use, implement proper model persistence and session management
- The API uses MLflow for experiment tracking and model versioning

## Testing

A comprehensive test suite is available at `/tests/test_training_endpoints.py` covering all endpoints with various scenarios including:
- Model training for both classification and regression
- Hyperparameter tuning
- Predictions with different data types
- Session management
- Error handling

Run tests with:
```bash
pytest tests/test_training_endpoints.py -v
```

Current test status: **13/15 tests passing** ✅
