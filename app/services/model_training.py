"""
Model training service for machine learning operations with MLflow integration
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
import mlflow
import mlflow.sklearn

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from app.core.enums import ModelType


class ModelTrainingService:
    """Service for machine learning model training and evaluation with MLflow tracking"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_classifier = None
        self.training_history = []
        
        # Default hyperparameters for each model
        self.default_hyperparameters = {
            # Classification Models
            ModelType.LOGISTIC_REGRESSION: {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'random_state': 42
            },
            ModelType.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            ModelType.DECISION_TREE_CLASSIFIER: {
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            ModelType.SVC: {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            },
            ModelType.KNN_CLASSIFIER: {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            },
            ModelType.NAIVE_BAYES: {
                'var_smoothing': 1e-9
            },
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            ModelType.MLP_CLASSIFIER: {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 200,
                'random_state': 42
            },
            
            # Regression Models
            ModelType.LINEAR_REGRESSION: {},
            ModelType.RANDOM_FOREST_REGRESSOR: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            ModelType.DECISION_TREE_REGRESSOR: {
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            ModelType.SVR: {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            ModelType.KNN_REGRESSOR: {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            },
            ModelType.RIDGE: {
                'alpha': 1.0,
                'random_state': 42
            },
            ModelType.LASSO: {
                'alpha': 1.0,
                'random_state': 42
            },
            ModelType.ELASTIC_NET: {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': 42
            },
            ModelType.MLP_REGRESSOR: {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 200,
                'random_state': 42
            }
        }
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            self.default_hyperparameters[ModelType.XGBOOST_CLASSIFIER] = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            }
            self.default_hyperparameters[ModelType.XGBOOST_REGRESSOR] = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            }
            
        if LIGHTGBM_AVAILABLE:
            self.default_hyperparameters[ModelType.LIGHTGBM_CLASSIFIER] = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            }
            self.default_hyperparameters[ModelType.LIGHTGBM_REGRESSOR] = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            }
    
    def _get_model_instance(self, model_type: ModelType, hyperparameters: Dict[str, Any] = None) -> Any:
        """Get model instance with hyperparameters"""
        params = hyperparameters or self.default_hyperparameters.get(model_type, {})
        
        # Classification Models
        if model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(**params)
        elif model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            return RandomForestClassifier(**params)
        elif model_type == ModelType.DECISION_TREE_CLASSIFIER:
            return DecisionTreeClassifier(**params)
        elif model_type == ModelType.SVC:
            return SVC(**params)
        elif model_type == ModelType.KNN_CLASSIFIER:
            return KNeighborsClassifier(**params)
        elif model_type == ModelType.NAIVE_BAYES:
            return GaussianNB(**params)
        elif model_type == ModelType.GRADIENT_BOOSTING_CLASSIFIER:
            return GradientBoostingClassifier(**params)
        elif model_type == ModelType.MLP_CLASSIFIER:
            return MLPClassifier(**params)
        elif model_type == ModelType.XGBOOST_CLASSIFIER and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(**params)
        elif model_type == ModelType.LIGHTGBM_CLASSIFIER and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(**params)
            
        # Regression Models
        elif model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegression(**params)
        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            return RandomForestRegressor(**params)
        elif model_type == ModelType.DECISION_TREE_REGRESSOR:
            return DecisionTreeRegressor(**params)
        elif model_type == ModelType.SVR:
            return SVR(**params)
        elif model_type == ModelType.KNN_REGRESSOR:
            return KNeighborsRegressor(**params)
        elif model_type == ModelType.RIDGE:
            return Ridge(**params)
        elif model_type == ModelType.LASSO:
            return Lasso(**params)
        elif model_type == ModelType.ELASTIC_NET:
            return ElasticNet(**params)
        elif model_type == ModelType.MLP_REGRESSOR:
            return MLPRegressor(**params)
        elif model_type == ModelType.XGBOOST_REGRESSOR and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(**params)
        elif model_type == ModelType.LIGHTGBM_REGRESSOR and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _is_classification_model(self, model_type: ModelType) -> bool:
        """Check if model is for classification"""
        classification_models = {
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST_CLASSIFIER,
            ModelType.DECISION_TREE_CLASSIFIER,
            ModelType.SVC,
            ModelType.KNN_CLASSIFIER,
            ModelType.NAIVE_BAYES,
            ModelType.GRADIENT_BOOSTING_CLASSIFIER,
            ModelType.MLP_CLASSIFIER,
            ModelType.XGBOOST_CLASSIFIER,
            ModelType.LIGHTGBM_CLASSIFIER
        }
        return model_type in classification_models
    
    def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_cross_validation: bool = True,
        cv_folds: int = 5,
        experiment_name: str = "easyml_experiments"
    ) -> Dict[str, Any]:
        """
        Train a machine learning model with MLflow tracking
        
        Args:
            data: Training dataset
            target_column: Name of target column
            model_type: Type of model to train
            test_size: Proportion of dataset to include in test split
            hyperparameters: Model hyperparameters (overrides defaults)
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            # Set MLflow experiment
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Validate inputs
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
                
                # Prepare features and target
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Determine if this is a classification or regression problem
                self.is_classifier = self._is_classification_model(model_type)
                
                # Get model instance with hyperparameters
                merged_params = self.default_hyperparameters.get(model_type, {}).copy()
                if hyperparameters:
                    merged_params.update(hyperparameters)
                
                self.model = self._get_model_instance(model_type, merged_params)
                self.model_type = model_type
                
                # Log hyperparameters to MLflow
                mlflow.log_params(merged_params)
                mlflow.log_param("model_type", model_type.value if hasattr(model_type, 'value') else str(model_type))
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("is_classifier", self.is_classifier)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, 
                    stratify=y if self.is_classifier else None
                )
                
                # Log dataset info
                mlflow.log_param("n_samples", len(data))
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_test", len(X_test))
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = self.model.predict(X_train)
                y_test_pred = self.model.predict(X_test)
                
                # Calculate metrics
                results = {
                    "model_type": model_type.value if hasattr(model_type, 'value') else str(model_type),
                    "hyperparameters": merged_params,
                    "is_classifier": self.is_classifier,
                    "n_samples": len(data),
                    "n_features": X.shape[1],
                    "test_size": test_size
                }
                
                if self.is_classifier:
                    # Classification metrics
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
                    results.update({
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "train_precision": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                        "train_recall": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                        "train_f1": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        "test_f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                    })
                    
                    # Log classification metrics
                    mlflow.log_metric("train_accuracy", train_accuracy)
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    mlflow.log_metric("test_precision", results["test_precision"])
                    mlflow.log_metric("test_recall", results["test_recall"])
                    mlflow.log_metric("test_f1", results["test_f1"])
                    
                    # Try to calculate ROC AUC if possible
                    try:
                        if hasattr(self.model, "predict_proba"):
                            y_test_proba = self.model.predict_proba(X_test)[:, 1]
                            test_auc = roc_auc_score(y_test, y_test_proba)
                            results["test_auc"] = test_auc
                            mlflow.log_metric("test_auc", test_auc)
                    except Exception:
                        pass  # Skip AUC if binary classification not applicable
                else:
                    # Regression metrics
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    results.update({
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "train_rmse": np.sqrt(train_mse),
                        "test_rmse": np.sqrt(test_mse),
                        "train_r2": train_r2,
                        "test_r2": test_r2
                    })
                    
                    # Log regression metrics
                    mlflow.log_metric("test_mae", test_mae)
                    mlflow.log_metric("test_mse", test_mse)
                    mlflow.log_metric("test_rmse", results["test_rmse"])
                    mlflow.log_metric("test_r2", test_r2)
                
                # Cross-validation if requested
                if use_cross_validation:
                    if self.is_classifier:
                        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy')
                        scoring_metric = 'accuracy'
                    else:
                        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2')
                        scoring_metric = 'r2'
                    
                    results.update({
                        "cv_scores": cv_scores.tolist(),
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "cv_scoring": scoring_metric
                    })
                    
                    mlflow.log_metric("cv_mean", cv_scores.mean())
                    mlflow.log_metric("cv_std", cv_scores.std())
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    self.model, 
                    "model",
                    registered_model_name=f"easyml_{model_type.value if hasattr(model_type, 'value') else str(model_type)}"
                )
                
                # Save training history
                self.training_history.append(results)
                
                return results
                
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type"""
        classification_models = [model.value for model in ModelType if self._is_classification_model(model)]
        regression_models = [model.value for model in ModelType if not self._is_classification_model(model)]
        
        return {
            "classification": classification_models,
            "regression": regression_models
        }
    
    def get_default_hyperparameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default hyperparameters for a model"""
        return self.default_hyperparameters.get(model_type, {})
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with trained model"""
        if self.model is None:
            return {"error": "No model has been trained yet"}
        
        try:
            predictions = self.model.predict(data)
            
            result = {
                "predictions": predictions.tolist(),
                "model_type": self.model_type.value if hasattr(self.model_type, 'value') else str(self.model_type),
                "is_classifier": self.is_classifier
            }
            
            # Add probabilities for classifiers if available
            if self.is_classifier and hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(data)
                    result["probabilities"] = probabilities.tolist()
                    if hasattr(self.model, 'classes_'):
                        result["classes"] = [str(c) for c in self.model.classes_.tolist()]
                except Exception:
                    pass  # Skip probabilities if not available
            
            return result
            
        except Exception as e:
            return {"error": f"Error making predictions: {str(e)}"}
    
    
    def get_model_serialized(self):
        """Get model as base64 encoded data"""
        if self.model is None:
            return {"error": "No model has been trained yet"}
        
        try:
            import base64
            import io
            
            # Serialize model to bytes
            buffer = io.BytesIO()
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_classifier': self.is_classifier
            }, buffer)
            
            # Encode as base64
            buffer.seek(0)
            model_bytes = buffer.getvalue()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            return {
                "model_data": model_b64,
                "model_type": self.model_type.value if hasattr(self.model_type, 'value') else str(self.model_type),
                "is_classifier": self.is_classifier,
                "size_bytes": len(model_bytes)
            }
        except Exception as e:
            return {"error": f"Error serializing model: {str(e)}"}
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            return {"error": "No model has been trained yet"}
        
        try:
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_classifier': self.is_classifier
            }, filepath)
            
            return {
                "success": True,
                "filepath": filepath,
                "message": "Model saved successfully"
            }
        except Exception as e:
            return {"error": f"Error saving model: {str(e)}"}
    
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_classifier = model_data['is_classifier']
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def hyperparameter_tuning(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        param_grid: Dict[str, List],
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        experiment_name: str = "easyml_hyperparameter_tuning"
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV with MLflow tracking
        
        Args:
            data: Training dataset
            target_column: Name of target column
            model_type: Type of model to tune
            param_grid: Grid of parameters to search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary with best parameters and results
        """
        try:
            # Set MLflow experiment
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Prepare data
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Determine problem type
                self.is_classifier = self._is_classification_model(model_type)
                
                # Get base model
                base_model = self._get_model_instance(model_type)
                default_scoring = 'accuracy' if self.is_classifier else 'r2'
                scoring = scoring or default_scoring
                
                # Log parameters
                mlflow.log_param("model_type", model_type.value if hasattr(model_type, 'value') else str(model_type))
                mlflow.log_param("cv_folds", cv_folds)
                mlflow.log_param("scoring", scoring)
                mlflow.log_param("n_samples", len(data))
                mlflow.log_param("n_features", X.shape[1])
                
                # Perform grid search
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X, y)
                
                # Log best parameters and score
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                
                # Log the best model
                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    "best_model",
                    registered_model_name=f"easyml_{model_type.value if hasattr(model_type, 'value') else str(model_type)}_tuned"
                )
                
                results = {
                    "best_params": grid_search.best_params_,
                    "best_score": grid_search.best_score_,
                    "cv_results": {
                        "mean_test_score": grid_search.cv_results_['mean_test_score'].tolist(),
                        "std_test_score": grid_search.cv_results_['std_test_score'].tolist(),
                        "params": grid_search.cv_results_['params']
                    },
                    "model_type": model_type.value if hasattr(model_type, 'value') else str(model_type),
                    "scoring": scoring,
                    "cv_folds": cv_folds
                }
                
                # Update model with best parameters
                self.model = grid_search.best_estimator_
                self.model_type = model_type
                
                return results
                
        except Exception as e:
            raise Exception(f"Error during hyperparameter tuning: {str(e)}")
    
    def predict_with_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using trained model
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dictionary with predictions and probabilities (if applicable)
        """
        if self.model is None:
            return {"error": "No trained model available"}
        
        try:
            predictions = self.model.predict(data)
            
            result = {
                "predictions": predictions.tolist(),
                "model_type": self.model_type.value if self.model_type else None,
                "is_classifier": self.is_classifier
            }
            
            # Add prediction probabilities for classifiers
            if self.is_classifier and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data)
                result["probabilities"] = probabilities.tolist()
                result["classes"] = self.model.classes_.tolist()
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def save_model(self, file_path: str) -> Dict[str, Any]:
        """Save trained model to file"""
        if self.model is None:
            return {"error": "No trained model to save"}
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_classifier': self.is_classifier
            }, file_path)
            
            # Handle model_type properly
            model_type_str = None
            if self.model_type:
                if hasattr(self.model_type, 'value'):
                    model_type_str = self.model_type.value
                else:
                    model_type_str = str(self.model_type)
            
            return {
                "success": True,
                "file_path": file_path,
                "model_type": model_type_str
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def load_model(self, file_path: str) -> Dict[str, Any]:
        """Load trained model from file"""
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.is_classifier = model_data['is_classifier']
            
            # Handle model_type properly
            model_type_str = None
            if self.model_type:
                if hasattr(self.model_type, 'value'):
                    model_type_str = self.model_type.value
                else:
                    model_type_str = str(self.model_type)
            
            return {
                "success": True,
                "model_type": model_type_str,
                "is_classifier": self.is_classifier
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_serialized(self) -> Dict[str, Any]:
        """Get model as base64 encoded string"""
        if self.model is None:
            return {"error": "No trained model available"}
        
        try:
            buffer = io.BytesIO()
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_classifier': self.is_classifier
            }, buffer)
            
            model_bytes = buffer.getvalue()
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Handle model_type properly
            model_type_str = None
            if self.model_type:
                if hasattr(self.model_type, 'value'):
                    model_type_str = self.model_type.value
                else:
                    model_type_str = str(self.model_type)
            
            return {
                "model_data": model_base64,
                "model_type": model_type_str,
                "is_classifier": self.is_classifier,
                "size_bytes": len(model_bytes)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _is_classification_problem(self, y: pd.Series) -> bool:
        """Determine if the problem is classification or regression"""
        # Check if target is categorical or has limited unique values
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        unique_values = y.nunique()
        total_values = len(y)
        
        # If less than 10 unique values or less than 5% unique values, treat as classification
        return unique_values < 10 or (unique_values / total_values) < 0.05
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""
        metrics = {}
        
        if self.is_classifier:
            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # Handle multiclass vs binary classification
            average = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
            
            metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    if hasattr(self.model, 'predict_proba'):
                        y_prob = self.model.predict_proba(y_true.index)[:, 1]
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                except:
                    pass
        else:
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def _get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get history of all training sessions"""
        return self.training_history.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        if self.model is None:
            return {"error": "No trained model available"}
        
        return {
            "model_type": self.model_type.value if hasattr(self.model_type, 'value') else str(self.model_type),
            "is_classifier": self.is_classifier,
            "parameters": self.model.get_params(),
            "has_feature_importance": hasattr(self.model, 'feature_importances_'),
            "has_predict_proba": hasattr(self.model, 'predict_proba')
        }
