"""
Model training endpoints with DVC integration
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from app.services.enhanced_model_training import EnhancedModelTrainingService
from app.services.file_service import FileService
from app.services.preprocessing import PreprocessingService
from app.schemas.schemas import ModelTrainingRequest, ModelTrainingResponse, PredictionRequest, PredictionResponse, HyperparameterTuningRequest, HyperparameterTuningResponse
from app.core.enums import ModelType
from app.core.auth import get_current_active_user
from app.core.database import get_db
from app.models.sql_models import User, Project

router = APIRouter()
file_service = FileService()

# Store active training services (in production, use proper session management)
training_services: Dict[str, EnhancedModelTrainingService] = {}


async def get_data_dependency(file_id: str):
    """Dependency to load data from file"""
    result = await file_service.load_data(file_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    return result["data"]


def get_or_create_training_service(
    session_id: str,
    user: Optional[User] = None,
    project: Optional[Project] = None,
    db_session: Optional[Session] = None
) -> EnhancedModelTrainingService:
    """Get or create a training service for a session"""
    if session_id not in training_services:
        training_services[session_id] = EnhancedModelTrainingService(
            session_id=session_id,
            user=user,
            project=project,
            db_session=db_session
        )
    else:
        # Update existing service with current context
        service = training_services[session_id]
        service.user = user
        service.project = project
        service.db_session = db_session
    return training_services[session_id]


@router.post("/projects/{project_id}/train", response_model=ModelTrainingResponse)
async def train_model(
    project_id: str,
    request: ModelTrainingRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Train a machine learning model with automatic DVC versioning
    
    - **project_id**: ID of the project this model belongs to
    - **file_id**: ID of the uploaded data file
    - **target_column**: Name of the target/label column
    - **model_type**: Type of ML model to train
    - **test_size**: Proportion of data for testing (0.1-0.5)
    - **preprocessing_operations**: Optional preprocessing to apply before training
    - **hyperparameters**: Optional model hyperparameters
    - **use_cross_validation**: Whether to use cross-validation
    - **cv_folds**: Number of cross-validation folds
    - **session_id**: Session identifier for model management
    - **auto_version**: Whether to automatically version the model with DVC
    """
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.is_active == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
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
    
    # Create enhanced training service using the session-based approach
    print(f"🔍 ENDPOINT DEBUG: Creating enhanced training service for session {request.session_id}")
    training_service = get_or_create_training_service(
        session_id=request.session_id,
        user=current_user,
        project=project,
        db_session=db
    )
    print(f"🔍 ENDPOINT DEBUG: Service type: {type(training_service)}")
    
    # Train model with persistence and DVC integration
    print(f"🔍 ENDPOINT DEBUG: Calling train_model_with_persistence...")
    result = await training_service.train_model_with_persistence(
        data=data,
        target_column=request.target_column,
        model_type=request.model_type,
        test_size=request.test_size,
        hyperparameters=request.hyperparameters,
        use_cross_validation=request.use_cross_validation,
        cv_folds=request.cv_folds,
        auto_version=getattr(request, 'auto_version', True)  # Default to True for automatic versioning
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    print(f"🔍 ENDPOINT DEBUG: Training completed! Result keys: {list(result.keys())}")
    
    # Construct train and test metrics dictionaries
    train_metrics = {}
    test_metrics = {}
    
    for key, value in result.items():
        if key.startswith("train_"):
            train_metrics[key.replace("train_", "")] = value
        elif key.startswith("test_"):
            test_metrics[key.replace("test_", "")] = value
    
    return ModelTrainingResponse(
        success=True,
        model_type=result["model_type"],
        is_classifier=result["is_classifier"],
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_scores=result.get("cv_scores"),
        cv_mean=result.get("cv_mean"),
        cv_std=result.get("cv_std"),
        feature_importance=result.get("feature_importance", {}),
        hyperparameters=result["hyperparameters"],
        data_shape=(result["n_samples"], result["n_features"]),
        session_id=request.session_id
    )


@router.post("/hyperparameter-tuning", response_model=HyperparameterTuningResponse)
async def tune_hyperparameters(request: HyperparameterTuningRequest):
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
    data = await get_data_dependency(request.file_id)
    
    # Apply preprocessing if specified
    if request.preprocessing_operations:
        preprocess_service = PreprocessingService(data)
        preprocess_result = preprocess_service.apply_preprocessing(
            request.preprocessing_operations, request.is_categorical
        )
        
        if preprocess_result["errors"]:
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing errors: {'; '.join(preprocess_result['errors'])}"
            )
        
        data = preprocess_service.get_processed_data()
    
    # Get or create training service
    training_service = get_or_create_training_service(request.session_id)
    
    # Perform hyperparameter tuning
    result = training_service.hyperparameter_tuning(
        data=data,
        target_column=request.target_column,
        model_type=request.model_type,
        param_grid=request.param_grid,
        cv_folds=request.cv_folds,
        scoring=request.scoring
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return HyperparameterTuningResponse(
        success=True,
        **result,
        session_id=request.session_id
    )


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
        ModelType.RIDGE: {
            "description": "Linear regression with L2 regularization",
            "type": "regression",
            "hyperparameters": ["alpha", "fit_intercept"],
            "pros": ["Handles multicollinearity", "Reduces overfitting"],
            "cons": ["Biased estimates", "All features retained"]
        },
        ModelType.LASSO: {
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
        ModelType.DECISION_TREE_REGRESSOR: {
            "description": "Single decision tree for regression",
            "type": "regression",
            "hyperparameters": ["max_depth", "min_samples_split", "min_samples_leaf"],
            "pros": ["Interpretable", "Handles non-linear relationships", "No preprocessing needed"],
            "cons": ["Prone to overfitting", "Unstable"]
        },
        ModelType.SVR: {
            "description": "Support Vector Machine for regression",
            "type": "regression",
            "hyperparameters": ["C", "kernel", "gamma", "epsilon"],
            "pros": ["Effective in high dimensions", "Memory efficient", "Versatile"],
            "cons": ["No probabilistic output", "Sensitive to scaling"]
        },
        ModelType.KNN_REGRESSOR: {
            "description": "K-Nearest Neighbors for regression",
            "type": "regression",
            "hyperparameters": ["n_neighbors", "weights", "algorithm"],
            "pros": ["Simple", "No assumptions about data", "Works with small datasets"],
            "cons": ["Computationally expensive", "Sensitive to irrelevant features"]
        },
        ModelType.ELASTIC_NET: {
            "description": "Linear regression with L1 and L2 regularization",
            "type": "regression",
            "hyperparameters": ["alpha", "l1_ratio", "fit_intercept"],
            "pros": ["Combines Ridge and Lasso", "Handles correlated features", "Feature selection"],
            "cons": ["Requires tuning", "Less interpretable than simple linear"]
        },
        ModelType.MLP_REGRESSOR: {
            "description": "Multi-layer Perceptron neural network for regression",
            "type": "regression",
            "hyperparameters": ["hidden_layer_sizes", "activation", "solver", "alpha"],
            "pros": ["Handles non-linear relationships", "Flexible architecture"],
            "cons": ["Black box", "Requires large datasets", "Sensitive to scaling"]
        },
        ModelType.XGBOOST_REGRESSOR: {
            "description": "Extreme Gradient Boosting for regression",
            "type": "regression",
            "hyperparameters": ["n_estimators", "max_depth", "learning_rate", "subsample"],
            "pros": ["High performance", "Built-in regularization", "Handles missing values"],
            "cons": ["Many hyperparameters", "Can overfit", "Requires tuning"]
        },
        ModelType.LIGHTGBM_REGRESSOR: {
            "description": "Light Gradient Boosting Machine for regression",
            "type": "regression",
            "hyperparameters": ["n_estimators", "max_depth", "learning_rate", "num_leaves"],
            "pros": ["Fast training", "Low memory usage", "High accuracy"],
            "cons": ["Can overfit small datasets", "Many hyperparameters"]
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
        },
        ModelType.DECISION_TREE_CLASSIFIER: {
            "description": "Single decision tree for classification",
            "type": "classification",
            "hyperparameters": ["max_depth", "min_samples_split", "min_samples_leaf"],
            "pros": ["Interpretable", "Handles non-linear relationships", "No preprocessing needed"],
            "cons": ["Prone to overfitting", "Unstable"]
        },
        ModelType.KNN_CLASSIFIER: {
            "description": "K-Nearest Neighbors for classification",
            "type": "classification",
            "hyperparameters": ["n_neighbors", "weights", "algorithm"],
            "pros": ["Simple", "No assumptions about data", "Works with small datasets"],
            "cons": ["Computationally expensive", "Sensitive to irrelevant features"]
        },
        ModelType.MLP_CLASSIFIER: {
            "description": "Multi-layer Perceptron neural network for classification",
            "type": "classification",
            "hyperparameters": ["hidden_layer_sizes", "activation", "solver", "alpha"],
            "pros": ["Handles non-linear relationships", "Flexible architecture"],
            "cons": ["Black box", "Requires large datasets", "Sensitive to scaling"]
        },
        ModelType.XGBOOST_CLASSIFIER: {
            "description": "Extreme Gradient Boosting for classification",
            "type": "classification",
            "hyperparameters": ["n_estimators", "max_depth", "learning_rate", "subsample"],
            "pros": ["High performance", "Built-in regularization", "Handles missing values"],
            "cons": ["Many hyperparameters", "Can overfit", "Requires tuning"]
        },
        ModelType.LIGHTGBM_CLASSIFIER: {
            "description": "Light Gradient Boosting Machine for classification",
            "type": "classification",
            "hyperparameters": ["n_estimators", "max_depth", "learning_rate", "num_leaves"],
            "pros": ["Fast training", "Low memory usage", "High accuracy"],
            "cons": ["Can overfit small datasets", "Many hyperparameters"]
        }
    }
    
    return {
        "success": True,
        "models": model_info,
        "regression_models": [model.value for model in ModelType if "REGRESSOR" in model.name or model.name in ["LINEAR_REGRESSION", "RIDGE", "LASSO", "ELASTIC_NET", "SVR"]],
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


@router.get("/models/available")
async def get_available_models():
    """
    Get list of all available machine learning models
    """
    training_service = EnhancedModelTrainingService()
    models = training_service.get_available_models()
    
    return {
        "success": True,
        "models": models
    }


@router.get("/models/{model_type}/hyperparameters")
async def get_model_hyperparameters(model_type: ModelType):
    """
    Get default hyperparameters for a specific model type
    """
    training_service = EnhancedModelTrainingService()
    hyperparameters = training_service.get_default_hyperparameters(model_type)
    
    return {
        "success": True,
        "model_type": model_type.value,
        "default_hyperparameters": hyperparameters
    }
