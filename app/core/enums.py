"""
Enumerations for EasyML application
"""
from enum import Enum


class PlotType(str, Enum):
    """Available plot types for data visualization"""
    
    # Basic plots (already implemented)
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    CORRELATION_MATRIX = "correlation_matrix"
    BOXPLOT = "boxplot"
    PAIRPLOT = "pairplot"
    COUNTPLOT = "countplot"
    
    # Numeric feature plots
    VIOLIN = "violin"
    KDE = "kde"
    
    # Categorical feature plots
    PIE = "pie"
    TARGET_MEAN = "target_mean"
    STACKED_BAR = "stacked_bar"
    CHI_SQUARED_HEATMAP = "chi_squared_heatmap"
    
    # Mixed & ML use-case plots
    PCA_SCATTER = "pca_scatter"
    CLASS_IMBALANCE = "class_imbalance"
    LEARNING_CURVE = "learning_curve"


class PreprocessingOption(str, Enum):
    """Available preprocessing options"""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    ENCODING = "encoding"
    IMPUTATION = "imputation"
    FEATURE_SELECTION = "feature_selection"
    SCALING = "scaling"


class ModelType(str, Enum):
    """Available machine learning models"""
    # Classification Models
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    SVC = "svc"  # Support Vector Classifier
    KNN_CLASSIFIER = "knn_classifier"
    NAIVE_BAYES = "naive_bayes"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    MLP_CLASSIFIER = "mlp_classifier"  # Multi-layer Perceptron
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    
    # Regression Models
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    SVR = "svr"  # Support Vector Regressor
    KNN_REGRESSOR = "knn_regressor"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    MLP_REGRESSOR = "mlp_regressor"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"


class DatasetMode(str, Enum):
    """Dataset modes for training/testing"""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    CREATED = "created"
    UPLOADED = "uploaded"
    PREPROCESSED = "preprocessed"
    TRAINED = "trained"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Machine learning task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class MetricType(str, Enum):
    """Available metrics for model evaluation"""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    
    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    MAPE = "mape"
    
    # Clustering metrics
    SILHOUETTE_SCORE = "silhouette_score"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"


class DeploymentEnvironment(str, Enum):
    """Available deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStrategy(str, Enum):
    """Available deployment strategies"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"
