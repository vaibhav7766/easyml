"""
Model training service for machine learning operations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import io
import base64
from typing import Dict, Any, List, Optional, Tuple

from app.core.enums import ModelType


class ModelTrainingService:
    """Service for machine learning model training and evaluation"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_classifier = None
        self.training_history = []
        
        # Define available models
        self.regression_models = {
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RIDGE_REGRESSION: Ridge(),
            ModelType.LASSO_REGRESSION: Lasso(),
            ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor(random_state=42),
            ModelType.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor(random_state=42),
            ModelType.SVR: SVR(),
            ModelType.KNN_REGRESSOR: KNeighborsRegressor(),
            ModelType.DECISION_TREE_REGRESSOR: DecisionTreeRegressor(random_state=42)
        }
        
        self.classification_models = {
            ModelType.LOGISTIC_REGRESSION: LogisticRegression(random_state=42),
            ModelType.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier(random_state=42),
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier(random_state=42),
            ModelType.SVC: SVC(random_state=42),
            ModelType.KNN_CLASSIFIER: KNeighborsClassifier(),
            ModelType.DECISION_TREE_CLASSIFIER: DecisionTreeClassifier(random_state=42),
            ModelType.NAIVE_BAYES: GaussianNB()
        }
    
    def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_cross_validation: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            data: Training dataset
            target_column: Name of target column
            model_type: Type of model to train
            test_size: Proportion of dataset to include in test split
            hyperparameters: Model hyperparameters
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            # Validate inputs
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Determine if this is a classification or regression problem
            self.is_classifier = self._is_classification_problem(y)
            
            # Select appropriate model
            if self.is_classifier:
                if model_type not in self.classification_models:
                    raise ValueError(f"Model type {model_type} not available for classification")
                self.model = self.classification_models[model_type]
            else:
                if model_type not in self.regression_models:
                    raise ValueError(f"Model type {model_type} not available for regression")
                self.model = self.regression_models[model_type]
            
            self.model_type = model_type
            
            # Apply hyperparameters if provided
            if hyperparameters:
                self.model.set_params(**hyperparameters)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if self.is_classifier else None
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = None
            if use_cross_validation:
                scoring = 'accuracy' if self.is_classifier else 'r2'
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring=scoring)
            
            # Feature importance (if available)
            feature_importance = self._get_feature_importance(X.columns)
            
            # Store training history
            training_result = {
                "model_type": model_type.value,
                "is_classifier": self.is_classifier,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
                "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                "feature_importance": feature_importance,
                "hyperparameters": self.model.get_params(),
                "data_shape": data.shape,
                "target_column": target_column,
                "test_size": test_size
            }
            
            self.training_history.append(training_result)
            
            return training_result
            
        except Exception as e:
            return {
                "error": str(e),
                "model_type": model_type.value if model_type else None,
                "success": False
            }
    
    def hyperparameter_tuning(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        param_grid: Dict[str, List],
        cv_folds: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            data: Training dataset
            target_column: Name of target column
            model_type: Type of model to tune
            param_grid: Grid of parameters to search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary with best parameters and results
        """
        try:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Determine problem type
            self.is_classifier = self._is_classification_problem(y)
            
            # Select model
            if self.is_classifier:
                if model_type not in self.classification_models:
                    raise ValueError(f"Model type {model_type} not available for classification")
                base_model = self.classification_models[model_type]
                default_scoring = 'accuracy'
            else:
                if model_type not in self.regression_models:
                    raise ValueError(f"Model type {model_type} not available for regression")
                base_model = self.regression_models[model_type]
                default_scoring = 'r2'
            
            # Use provided scoring or default
            scoring = scoring or default_scoring
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            
            grid_search.fit(X, y)
            
            # Store best model
            self.model = grid_search.best_estimator_
            self.model_type = model_type
            
            return {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_estimator": str(grid_search.best_estimator_),
                "cv_results": {
                    "mean_test_scores": grid_search.cv_results_['mean_test_score'].tolist(),
                    "mean_train_scores": grid_search.cv_results_['mean_train_score'].tolist(),
                    "params": grid_search.cv_results_['params']
                },
                "model_type": model_type.value,
                "scoring": scoring,
                "cv_folds": cv_folds
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
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
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_classifier': self.is_classifier
            }, file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "model_type": self.model_type.value if self.model_type else None
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
            
            return {
                "success": True,
                "model_type": self.model_type.value if self.model_type else None,
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
            
            return {
                "model_data": model_base64,
                "model_type": self.model_type.value if self.model_type else None,
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
            "model_type": self.model_type.value if self.model_type else None,
            "is_classifier": self.is_classifier,
            "parameters": self.model.get_params(),
            "has_feature_importance": hasattr(self.model, 'feature_importances_'),
            "has_predict_proba": hasattr(self.model, 'predict_proba')
        }
