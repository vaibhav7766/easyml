"""
EasyML Application Package
A comprehensive no-code machine learning automation platform with full CI/CD integration
"""

__version__ = "1.0.0"
__author__ = "EasyML Team"
__description__ = "No-code machine learning automation platform with CI/CD pipeline"
__license__ = "MIT"

# Application metadata
APP_NAME = "EasyML"
API_VERSION = "v1"
SUPPORTED_FILE_FORMATS = [".csv", ".xlsx", ".json", ".parquet", ".tsv"]
SUPPORTED_ML_MODELS = [
    "random_forest", "xgboost", "lightgbm", "linear_regression", 
    "logistic_regression", "svm", "neural_network", "decision_tree"
]

# Feature flags
FEATURES = {
    "data_upload": True,
    "data_visualization": True, 
    "data_preprocessing": True,
    "model_training": True,
    "model_deployment": True,
    "dvc_integration": True,
    "mlflow_tracking": True,
    "automated_pipelines": True,
    "containerized_deployment": True,
    "real_time_predictions": True,
    "model_versioning": True,
    "automated_testing": True
}
