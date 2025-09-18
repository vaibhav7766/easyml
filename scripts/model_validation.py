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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['model_validation']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_model_and_info(model_dir):
    """Load trained model and training info"""
    try:
        # Load model
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        
        # Load training info
        info_path = os.path.join(model_dir, 'training_info.json')
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        
        # Load evaluation results
        eval_path = os.path.join(model_dir.replace('models', 'evaluation'), 'evaluation_results.json')
        evaluation_results = None
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                evaluation_results = json.load(f)
        
        logger.info(f"Loaded model and info successfully")
        return model, training_info, evaluation_results
    except Exception as e:
        logger.error(f"Error loading model and info: {e}")
        raise

def validate_performance(evaluation_results, threshold):
    """Validate model performance against threshold"""
    logger.info("Validating model performance...")
    
    if not evaluation_results:
        logger.error("No evaluation results available")
        return False
    
    metrics = evaluation_results.get('metrics', {})
    task_type = evaluation_results.get('task_type', 'classification')
    
    # Choose primary metric based on task type
    if task_type == 'classification':
        primary_metric = 'f1_score'
        if primary_metric not in metrics:
            primary_metric = 'accuracy'
    else:
        primary_metric = 'r2_score'
        if primary_metric not in metrics:
            primary_metric = 'mse'
    
    if primary_metric not in metrics:
        logger.error(f"Primary metric {primary_metric} not found in evaluation results")
        return False
    
    performance_value = metrics[primary_metric]
    
    # For MSE, lower is better
    if primary_metric == 'mse':
        meets_threshold = performance_value <= threshold
    else:
        meets_threshold = performance_value >= threshold
    
    logger.info(f"Performance validation: {primary_metric} = {performance_value:.4f}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Meets threshold: {meets_threshold}")
    
    return meets_threshold

def detect_data_drift(original_data, new_data, threshold=0.1):
    """Detect data drift between original and new data"""
    logger.info("Detecting data drift...")
    
    try:
        from scipy import stats
        
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        drift_detected = False
        drift_results = {}
        
        for col in numeric_cols:
            if col in new_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(original_data[col], new_data[col])
                
                # Drift detected if p-value < threshold
                column_drift = p_value < threshold
                drift_detected = drift_detected or column_drift
                
                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': column_drift
                }
                
                if column_drift:
                    logger.warning(f"Data drift detected in column {col}: p-value = {p_value:.4f}")
                else:
                    logger.info(f"No drift in column {col}: p-value = {p_value:.4f}")
        
        return drift_detected, drift_results
        
    except ImportError:
        logger.warning("scipy not available, skipping data drift detection")
        return False, {}
    except Exception as e:
        logger.error(f"Error in data drift detection: {e}")
        return False, {}

def validate_model_explainability(model, X_sample, feature_names, output_dir):
    """Validate model explainability"""
    logger.info("Validating model explainability...")
    
    try:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Check if any features have very high importance (potential overfitting)
            max_importance = np.max(importance)
            high_importance_threshold = 0.8
            
            if max_importance > high_importance_threshold:
                logger.warning(f"Very high feature importance detected: {max_importance:.3f}")
                dominant_feature = feature_names[np.argmax(importance)]
                logger.warning(f"Dominant feature: {dominant_feature}")
            
            # Check for features with zero importance
            zero_importance_features = [feature_names[i] for i, imp in enumerate(importance) if imp == 0]
            if zero_importance_features:
                logger.info(f"Features with zero importance: {zero_importance_features}")
            
            explainability_results = {
                'max_importance': max_importance,
                'dominant_feature': feature_names[np.argmax(importance)],
                'zero_importance_features': zero_importance_features,
                'feature_importance_available': True
            }
        else:
            logger.info("Model does not provide feature importance")
            explainability_results = {'feature_importance_available': False}
        
        return explainability_results
        
    except Exception as e:
        logger.error(f"Error in explainability validation: {e}")
        return {'error': str(e)}

def check_bias_fairness(model, X, y, sensitive_features=None):
    """Check for bias and fairness issues"""
    logger.info("Checking bias and fairness...")
    
    try:
        fairness_results = {}
        
        # If sensitive features are provided, check for bias
        if sensitive_features:
            for feature in sensitive_features:
                if feature in X.columns:
                    unique_values = X[feature].unique()
                    
                    if len(unique_values) <= 10:  # Categorical feature
                        predictions = model.predict(X)
                        
                        # Calculate performance by group
                        group_performance = {}
                        for value in unique_values:
                            mask = X[feature] == value
                            if mask.sum() > 0:
                                group_pred = predictions[mask]
                                group_true = y[mask]
                                
                                if len(np.unique(y)) <= 10:  # Classification
                                    group_acc = accuracy_score(group_true, group_pred)
                                    group_performance[str(value)] = group_acc
                                else:  # Regression
                                    group_r2 = r2_score(group_true, group_pred)
                                    group_performance[str(value)] = group_r2
                        
                        # Check for significant performance differences
                        if len(group_performance) > 1:
                            performances = list(group_performance.values())
                            performance_diff = max(performances) - min(performances)
                            
                            fairness_results[feature] = {
                                'group_performance': group_performance,
                                'performance_difference': performance_diff,
                                'potential_bias': performance_diff > 0.1  # Threshold for bias
                            }
                            
                            if performance_diff > 0.1:
                                logger.warning(f"Potential bias detected for feature {feature}: "
                                             f"performance difference = {performance_diff:.3f}")
        
        # Overall prediction distribution analysis
        predictions = model.predict(X)
        pred_stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions))
        }
        
        fairness_results['prediction_statistics'] = pred_stats
        
        return fairness_results
        
    except Exception as e:
        logger.error(f"Error in bias/fairness check: {e}")
        return {'error': str(e)}

def validate_model(model_dir, validation_data_path, output_dir, original_data_path=None):
    """Main model validation function"""
    logger.info("Starting model validation...")
    
    # Load parameters
    params = load_parameters()
    
    # Load model and info
    model, training_info, evaluation_results = load_model_and_info(model_dir)
    
    # Load validation data
    validation_data = pd.read_csv(validation_data_path)
    
    # Prepare validation data
    target_column = training_info['target_column']
    feature_names = training_info['feature_names']
    task_type = training_info['task_type']
    
    X_val = validation_data[feature_names]
    y_val = validation_data[target_column]
    
    logger.info(f"Validation data shape: {X_val.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    validation_results = {
        'validation_timestamp': datetime.now().isoformat(),
        'model_type': training_info['model_type'],
        'task_type': task_type
    }
    
    # 1. Performance validation
    performance_threshold = params.get('performance_threshold', 0.8)
    performance_valid = validate_performance(evaluation_results, performance_threshold)
    validation_results['performance_validation'] = {
        'meets_threshold': performance_valid,
        'threshold': performance_threshold
    }
    
    # 2. Data drift detection
    if original_data_path and os.path.exists(original_data_path):
        original_data = pd.read_csv(original_data_path)
        drift_detected, drift_results = detect_data_drift(
            original_data[feature_names], 
            X_val,
            threshold=0.05
        )
        validation_results['data_drift'] = {
            'drift_detected': drift_detected,
            'drift_results': drift_results
        }
    else:
        logger.info("Original data not provided, skipping drift detection")
    
    # 3. Explainability analysis
    if params.get('explainability_analysis', True):
        explainability_results = validate_model_explainability(
            model, X_val.head(100), feature_names, output_dir
        )
        validation_results['explainability'] = explainability_results
    
    # 4. Bias and fairness check
    if params.get('bias_fairness_check', True):
        # You can specify sensitive features here
        sensitive_features = []  # Add sensitive feature names if available
        fairness_results = check_bias_fairness(model, X_val, y_val, sensitive_features)
        validation_results['fairness'] = fairness_results
    
    # 5. Model performance on validation set
    val_predictions = model.predict(X_val)
    if task_type == 'classification':
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average='weighted')
        validation_results['validation_performance'] = {
            'accuracy': val_accuracy,
            'f1_score': val_f1
        }
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
    else:
        val_r2 = r2_score(y_val, val_predictions)
        val_mse = np.mean((y_val - val_predictions) ** 2)
        validation_results['validation_performance'] = {
            'r2_score': val_r2,
            'mse': val_mse
        }
        logger.info(f"Validation R2: {val_r2:.4f}")
        logger.info(f"Validation MSE: {val_mse:.4f}")
    
    # Overall validation status
    overall_valid = (
        performance_valid and
        not validation_results.get('data_drift', {}).get('drift_detected', False)
    )
    
    validation_results['overall_validation'] = {
        'model_valid': overall_valid,
        'validation_passed': overall_valid
    }
    
    # Save validation results
    results_path = os.path.join(output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("validation_strategy", params.get('validation_strategy', 'holdout'))
        mlflow.log_metric("model_valid", int(overall_valid))
        mlflow.log_metric("performance_valid", int(performance_valid))
        
        if 'validation_performance' in validation_results:
            for metric, value in validation_results['validation_performance'].items():
                mlflow.log_metric(f"val_{metric}", value)
        
        mlflow.log_artifact(results_path)
    
    logger.info(f"Model validation completed. Results saved to {output_dir}")
    logger.info(f"Overall validation status: {'PASSED' if overall_valid else 'FAILED'}")
    
    return validation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Validation Pipeline')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model')
    parser.add_argument('--validation-data', required=True, help='Validation data path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--original-data', help='Original training data path for drift detection')
    
    args = parser.parse_args()
    
    try:
        validate_model(args.model_dir, args.validation_data, args.output, args.original_data)
        logger.info("Model validation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Model validation pipeline failed: {e}")
        sys.exit(1)
