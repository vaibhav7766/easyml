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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['model_evaluation']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_model_and_data(model_dir, data_path):
    """Load trained model and test data"""
    try:
        # Load model
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        
        # Load training info
        info_path = os.path.join(model_dir, 'training_info.json')
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        
        # Load data
        data = pd.read_csv(data_path)
        
        logger.info(f"Loaded model and data successfully")
        return model, training_info, data
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        raise

def evaluate_classification_model(model, X_test, y_test, output_dir):
    """Evaluate classification model"""
    logger.info("Evaluating classification model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC curve for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            metrics['roc_auc'] = roc_auc
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
            
            # Plot Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, label='PR Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate ROC curve: {e}")
    
    return metrics, class_report, cm

def evaluate_regression_model(model, X_test, y_test, output_dir):
    """Evaluate regression model"""
    logger.info("Evaluating regression model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # Histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'))
    plt.close()
    
    return metrics, residuals

def feature_importance_analysis(model, feature_names, output_dir):
    """Analyze feature importance"""
    logger.info("Analyzing feature importance...")
    
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
        top_features = importance_df.head(20)  # Top 20 features
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        return None

def check_performance_thresholds(metrics, thresholds):
    """Check if model meets performance thresholds"""
    logger.info("Checking performance thresholds...")
    
    threshold_results = {}
    
    for metric, threshold in thresholds.items():
        if metric in metrics:
            meets_threshold = metrics[metric] >= threshold
            threshold_results[metric] = {
                'value': metrics[metric],
                'threshold': threshold,
                'meets_threshold': meets_threshold
            }
            
            if meets_threshold:
                logger.info(f"✓ {metric}: {metrics[metric]:.4f} >= {threshold}")
            else:
                logger.warning(f"✗ {metric}: {metrics[metric]:.4f} < {threshold}")
    
    return threshold_results

def evaluate_model(model_dir, data_path, output_dir):
    """Main model evaluation function"""
    logger.info("Starting model evaluation...")
    
    # Load parameters
    params = load_parameters()
    
    # Load model and data
    model, training_info, data = load_model_and_data(model_dir, data_path)
    
    # Prepare data
    target_column = training_info['target_column']
    feature_names = training_info['feature_names']
    task_type = training_info['task_type']
    
    X = data[feature_names]
    y = data[target_column]
    
    logger.info(f"Evaluation data shape: {X.shape}")
    logger.info(f"Task type: {task_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model based on task type
    if task_type == 'classification':
        metrics, class_report, confusion_mat = evaluate_classification_model(model, X, y, output_dir)
        
        # Save classification report
        report_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=2)
            
    else:  # regression
        metrics, residuals = evaluate_regression_model(model, X, y, output_dir)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(model, feature_names, output_dir)
    
    # Check performance thresholds
    thresholds = params.get('performance_thresholds', {})
    threshold_results = check_performance_thresholds(metrics, thresholds)
    
    # Save evaluation results
    evaluation_results = {
        'metrics': metrics,
        'threshold_results': threshold_results,
        'task_type': task_type,
        'model_type': training_info['model_type'],
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Log to MLflow
    with mlflow.start_run():
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(f"eval_{metric}", value)
        
        # Log threshold results
        for metric, result in threshold_results.items():
            mlflow.log_metric(f"threshold_{metric}_met", int(result['meets_threshold']))
        
        # Log artifacts
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.csv', '.json')):
                mlflow.log_artifact(os.path.join(output_dir, file))
    
    logger.info(f"Model evaluation completed. Results saved to {output_dir}")
    logger.info(f"Evaluation metrics: {metrics}")
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation Pipeline')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model')
    parser.add_argument('--data', required=True, help='Evaluation data path')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(args.model_dir, args.data, args.output)
        logger.info("Model evaluation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        sys.exit(1)
