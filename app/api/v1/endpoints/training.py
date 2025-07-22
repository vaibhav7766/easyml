"""
Model training endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any

from app.services.model_training import ModelTrainingService
from app.services.file_service import FileService
from app.services.preprocessing import PreprocessingService
from app.schemas.schemas import ModelTrainingRequest, ModelTrainingResponse, PredictionRequest, PredictionResponse
from app.core.enums import ModelType

router = APIRouter()
file_service = FileService()

# Store active training services (in production, use proper session management)
training_services: Dict[str, ModelTrainingService] = {}


async def get_data_dependency(file_id: str):
    """Dependency to load data from file"""
    result = await file_service.load_data(file_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    return result["data"]


def get_or_create_training_service(session_id: str) -> ModelTrainingService:
    """Get or create a training service for a session"""
    if session_id not in training_services:
        training_services[session_id] = ModelTrainingService()
    return training_services[session_id]


@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """
    Train a machine learning model
    
    - **file_id**: ID of the uploaded data file
    - **target_column**: Name of the target/label column
    - **model_type**: Type of ML model to train
    - **test_size**: Proportion of data for testing (0.1-0.5)
    - **preprocessing_operations**: Optional preprocessing to apply before training
    - **hyperparameters**: Optional model hyperparameters
    - **use_cross_validation**: Whether to use cross-validation
    - **cv_folds**: Number of cross-validation folds
    - **session_id**: Session identifier for model management
    """
    # Load data
    data = await get_data_dependency(request.file_id)
    
    # Apply preprocessing if specified
    if request.preprocessing_operations:
        preprocess_service = PreprocessingService(data)
        preprocess_result = preprocess_service.apply_preprocessing(
            request.preprocessing_operations,
            request.is_categorical
        )
        
        if preprocess_result["errors"]:
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing errors: {'; '.join(preprocess_result['errors'])}"
            )
        
        data = preprocess_service.get_processed_data()
    
    # Get or create training service
    training_service = get_or_create_training_service(request.session_id)
    
    # Train model
    result = training_service.train_model(
        data=data,
        target_column=request.target_column,
        model_type=request.model_type,
        test_size=request.test_size,
        hyperparameters=request.hyperparameters,
        use_cross_validation=request.use_cross_validation,
        cv_folds=request.cv_folds
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return ModelTrainingResponse(
        success=True,
        model_type=result["model_type"],
        is_classifier=result["is_classifier"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        cv_scores=result.get("cv_scores"),
        cv_mean=result.get("cv_mean"),
        cv_std=result.get("cv_std"),
        feature_importance=result.get("feature_importance"),
        hyperparameters=result["hyperparameters"],
        data_shape=result["data_shape"],
        session_id=request.session_id
    )


@router.post("/hyperparameter-tuning")
async def tune_hyperparameters(
    file_id: str,
    target_column: str,
    model_type: ModelType,
    param_grid: Dict[str, List[Any]],
    cv_folds: int = 5,
    scoring: Optional[str] = None,
    session_id: str = "default",
    preprocessing_operations: Optional[Dict[str, str]] = None,
    is_categorical: bool = False
):
    """
    Perform hyperparameter tuning for a model
    
    - **file_id**: ID of the uploaded data file
    - **target_column**: Name of the target/label column
    - **model_type**: Type of ML model to tune
    - **param_grid**: Grid of parameters to search
    - **cv_folds**: Number of cross-validation folds
    - **scoring**: Scoring metric for optimization
    - **session_id**: Session identifier
    - **preprocessing_operations**: Optional preprocessing operations
    """
    # Load data
    data = await get_data_dependency(file_id)
    
    # Apply preprocessing if specified
    if preprocessing_operations:
        preprocess_service = PreprocessingService(data)
        preprocess_result = preprocess_service.apply_preprocessing(
            preprocessing_operations, is_categorical
        )
        
        if preprocess_result["errors"]:
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing errors: {'; '.join(preprocess_result['errors'])}"
            )
        
        data = preprocess_service.get_processed_data()
    
    # Get or create training service
    training_service = get_or_create_training_service(session_id)
    
    # Perform hyperparameter tuning
    result = training_service.hyperparameter_tuning(
        data=data,
        target_column=target_column,
        model_type=model_type,
        param_grid=param_grid,
        cv_folds=cv_folds,
        scoring=scoring
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "success": True,
        **result,
        "session_id": session_id
    }


@router.post("/predict", response_model=PredictionResponse)
async def make_predictions(request: PredictionRequest):
    """
    Make predictions using a trained model
    
    - **file_id**: ID of the data file for prediction
    - **session_id**: Session identifier with the trained model
    """
    # Load data
    data = await get_data_dependency(request.file_id)
    
    # Get training service
    if request.session_id not in training_services:
        raise HTTPException(status_code=404, detail="No trained model found for this session")
    
    training_service = training_services[request.session_id]
    
    # Make predictions
    result = training_service.predict(data)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return PredictionResponse(
        success=True,
        predictions=result["predictions"],
        model_type=result.get("model_type"),
        is_classifier=result.get("is_classifier", False),
        probabilities=result.get("probabilities"),
        classes=result.get("classes"),
        session_id=request.session_id
    )


@router.get("/model-info/{session_id}")
async def get_model_info(session_id: str):
    """
    Get information about a trained model
    
    - **session_id**: Session identifier
    """
    if session_id not in training_services:
        raise HTTPException(status_code=404, detail="No trained model found for this session")
    
    training_service = training_services[session_id]
    model_info = training_service.get_model_info()
    
    if "error" in model_info:
        raise HTTPException(status_code=400, detail=model_info["error"])
    
    return {
        "success": True,
        **model_info,
        "session_id": session_id
    }


@router.get("/training-history/{session_id}")
async def get_training_history(session_id: str):
    """
    Get training history for a session
    
    - **session_id**: Session identifier
    """
    if session_id not in training_services:
        raise HTTPException(status_code=404, detail="No training history found for this session")
    
    training_service = training_services[session_id]
    history = training_service.get_training_history()
    
    return {
        "success": True,
        "training_history": history,
        "total_trainings": len(history),
        "session_id": session_id
    }


@router.get("/available-models")
async def get_available_models():
    """
    Get list of available model types and their descriptions
    """
    model_info = {
        # Regression Models
        ModelType.LINEAR_REGRESSION: {
            "description": "Linear relationship modeling",
            "type": "regression",
            "hyperparameters": ["fit_intercept", "normalize"],
            "pros": ["Simple", "Interpretable", "Fast"],
            "cons": ["Assumes linearity", "Sensitive to outliers"]
        },
        ModelType.RIDGE_REGRESSION: {
            "description": "Linear regression with L2 regularization",
            "type": "regression",
            "hyperparameters": ["alpha", "fit_intercept"],
            "pros": ["Handles multicollinearity", "Reduces overfitting"],
            "cons": ["Biased estimates", "All features retained"]
        },
        ModelType.LASSO_REGRESSION: {
            "description": "Linear regression with L1 regularization",
            "type": "regression", 
            "hyperparameters": ["alpha", "fit_intercept", "max_iter"],
            "pros": ["Feature selection", "Sparse solutions"],
            "cons": ["Can be unstable", "Arbitrary feature selection"]
        },
        ModelType.RANDOM_FOREST_REGRESSOR: {
            "description": "Ensemble of decision trees for regression",
            "type": "regression",
            "hyperparameters": ["n_estimators", "max_depth", "min_samples_split"],
            "pros": ["Handles nonlinearity", "Feature importance", "Robust"],
            "cons": ["Less interpretable", "Can overfit"]
        },
        ModelType.GRADIENT_BOOSTING_REGRESSOR: {
            "description": "Sequential ensemble method",
            "type": "regression",
            "hyperparameters": ["n_estimators", "learning_rate", "max_depth"],
            "pros": ["High accuracy", "Handles mixed data types"],
            "cons": ["Sensitive to overfitting", "Requires tuning"]
        },
        
        # Classification Models
        ModelType.LOGISTIC_REGRESSION: {
            "description": "Linear model for classification",
            "type": "classification",
            "hyperparameters": ["C", "penalty", "solver"],
            "pros": ["Probabilistic output", "Interpretable", "Fast"],
            "cons": ["Assumes linearity", "Sensitive to outliers"]
        },
        ModelType.RANDOM_FOREST_CLASSIFIER: {
            "description": "Ensemble of decision trees for classification",
            "type": "classification",
            "hyperparameters": ["n_estimators", "max_depth", "min_samples_split"],
            "pros": ["Handles nonlinearity", "Feature importance", "Robust"],
            "cons": ["Less interpretable", "Can overfit"]
        },
        ModelType.GRADIENT_BOOSTING_CLASSIFIER: {
            "description": "Sequential ensemble for classification",
            "type": "classification",
            "hyperparameters": ["n_estimators", "learning_rate", "max_depth"],
            "pros": ["High accuracy", "Handles mixed data types"],
            "cons": ["Sensitive to overfitting", "Requires tuning"]
        },
        ModelType.SVC: {
            "description": "Support Vector Machine for classification",
            "type": "classification",
            "hyperparameters": ["C", "kernel", "gamma"],
            "pros": ["Effective in high dimensions", "Memory efficient"],
            "cons": ["No probabilistic output", "Sensitive to scaling"]
        },
        ModelType.NAIVE_BAYES: {
            "description": "Probabilistic classifier based on Bayes theorem",
            "type": "classification",
            "hyperparameters": [],
            "pros": ["Fast", "Good for text data", "Handles missing values"],
            "cons": ["Strong independence assumption", "Can be outperformed"]
        }
    }
    
    return {
        "success": True,
        "models": model_info,
        "regression_models": [model.value for model in ModelType if "REGRESSOR" in model.name or "REGRESSION" in model.name],
        "classification_models": [model.value for model in ModelType if "CLASSIFIER" in model.name or model.name in ["LOGISTIC_REGRESSION", "SVC", "NAIVE_BAYES"]]
    }


@router.post("/save-model/{session_id}")
async def save_model(session_id: str, filename: Optional[str] = None):
    """
    Save a trained model to file
    
    - **session_id**: Session identifier
    - **filename**: Optional custom filename
    """
    if session_id not in training_services:
        raise HTTPException(status_code=404, detail="No trained model found for this session")
    
    training_service = training_services[session_id]
    
    # Generate filename if not provided
    if not filename:
        model_info = training_service.get_model_info()
        if "error" in model_info:
            raise HTTPException(status_code=400, detail="Cannot get model info for filename generation")
        filename = f"model_{model_info['model_type']}_{session_id}.joblib"
    
    # Save model
    file_path = f"/tmp/{filename}"  # In production, use proper file storage
    result = training_service.save_model(file_path)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {
        "success": True,
        **result,
        "session_id": session_id
    }


@router.get("/export-model/{session_id}")
async def export_model(session_id: str):
    """
    Export a trained model as base64 encoded data
    
    - **session_id**: Session identifier
    """
    if session_id not in training_services:
        raise HTTPException(status_code=404, detail="No trained model found for this session")
    
    training_service = training_services[session_id]
    result = training_service.get_model_serialized()
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "success": True,
        **result,
        "session_id": session_id
    }


@router.delete("/session/{session_id}")
async def delete_training_session(session_id: str):
    """
    Delete a training session and its associated model
    
    - **session_id**: Session identifier to delete
    """
    if session_id not in training_services:
        raise HTTPException(status_code=404, detail="No training session found")
    
    del training_services[session_id]
    
    return {
        "success": True,
        "message": f"Training session {session_id} deleted successfully"
    }


@router.get("/sessions")
async def list_training_sessions():
    """
    List all active training sessions
    """
    sessions = []
    
    for session_id, service in training_services.items():
        model_info = service.get_model_info()
        history = service.get_training_history()
        
        sessions.append({
            "session_id": session_id,
            "has_model": "error" not in model_info,
            "model_info": model_info if "error" not in model_info else None,
            "training_count": len(history),
            "last_training": history[-1] if history else None
        })
    
    return {
        "success": True,
        "sessions": sessions,
        "total_sessions": len(sessions)
    }
