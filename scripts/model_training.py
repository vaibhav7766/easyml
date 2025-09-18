import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['model_training']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(data_path):
    """Load engineered data"""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def determine_task_type(y):
    """Determine if this is a classification or regression task"""
    unique_values = len(np.unique(y))
    
    if y.dtype == 'object' or unique_values <= 20:
        return 'classification'
    else:
        return 'regression'

def get_model(model_type, task_type, hyperparams):
    """Get model instance based on type and task"""
    logger.info(f"Creating {model_type} model for {task_type}")
    
    if model_type == 'random_forest':
        if task_type == 'classification':
            model = RandomForestClassifier(**hyperparams)
        else:
            model = RandomForestRegressor(**hyperparams)
    
    elif model_type == 'xgboost':
        if task_type == 'classification':
            model = xgb.XGBClassifier(**hyperparams)
        else:
            model = xgb.XGBRegressor(**hyperparams)
    
    elif model_type == 'lightgbm':
        if task_type == 'classification':
            model = lgb.LGBMClassifier(**hyperparams)
        else:
            model = lgb.LGBMRegressor(**hyperparams)
    
    elif model_type == 'neural_network':
        if task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=tuple(hyperparams.get('hidden_layers', [100])),
                activation=hyperparams.get('activation', 'relu'),
                alpha=hyperparams.get('dropout_rate', 0.0001),
                learning_rate_init=hyperparams.get('learning_rate', 0.001),
                max_iter=hyperparams.get('epochs', 200),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=hyperparams.get('early_stopping_patience', 10),
                random_state=hyperparams.get('random_state', 42)
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=tuple(hyperparams.get('hidden_layers', [100])),
                activation=hyperparams.get('activation', 'relu'),
                alpha=hyperparams.get('dropout_rate', 0.0001),
                learning_rate_init=hyperparams.get('learning_rate', 0.001),
                max_iter=hyperparams.get('epochs', 200),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=hyperparams.get('early_stopping_patience', 10),
                random_state=hyperparams.get('random_state', 42)
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def perform_cross_validation(model, X, y, cv_params, task_type):
    """Perform cross validation"""
    logger.info("Performing cross validation...")
    
    folds = cv_params.get('folds', 5)
    shuffle = cv_params.get('shuffle', True)
    random_state = cv_params.get('random_state', 42)
    
    if task_type == 'classification' and cv_params.get('stratified', True):
        cv = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def train_model(input_path, output_dir, target_column=None):
    """Main model training function"""
    logger.info("Starting model training...")
    
    # Load parameters
    params = load_parameters()
    
    # Load data
    data = load_data(input_path)
    
    # Separate features and target
    if target_column and target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        target_column = data.columns[-1]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Determine task type
    task_type = determine_task_type(y)
    logger.info(f"Detected task type: {task_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data for final evaluation
    test_size = 0.2
    random_state = params.get('random_state', 42)
    
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Get model
    model_type = params.get('model_type', 'random_forest')
    hyperparams = params.get('hyperparameters', {}).get(model_type, {})
    
    model = get_model(model_type, task_type, hyperparams)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_column)
        
        # Perform cross validation
        if params.get('cross_validation'):
            cv_scores = perform_cross_validation(
                model, X_train, y_train, 
                params['cross_validation'], 
                task_type
            )
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
        
        # Train model
        logger.info("Training final model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        if task_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import roc_auc_score, log_loss
            
            # Training metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train, average='weighted')
            train_recall = recall_score(y_train, y_pred_train, average='weighted')
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            
            # Test metrics
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)
            
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)
            
            # ROC AUC for binary classification
            if len(np.unique(y)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
                    mlflow.log_metric("test_roc_auc", test_roc_auc)
                except:
                    logger.warning("Could not calculate ROC AUC")
            
            metrics = {
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
            
        else:  # regression
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Training metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            
            # Test metrics
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_r2", train_r2)
            
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            
            metrics = {
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
        
        # Save model
        model_path = os.path.join(output_dir, 'model.joblib')
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save training info
        training_info = {
            'model_type': model_type,
            'task_type': task_type,
            'target_column': target_column,
            'hyperparameters': hyperparams,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'feature_names': list(X.columns)
        }
        
        info_path = os.path.join(output_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model training completed. Output saved to {output_dir}")
        logger.info(f"Test metrics: {metrics}")
        
        return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Pipeline')
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--target', help='Target column name')
    
    args = parser.parse_args()
    
    try:
        train_model(args.input, args.output, args.target)
        logger.info("Model training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        sys.exit(1)
