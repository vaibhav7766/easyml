from enum import Enum

class Plots(str, Enum):
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

class Options(str, Enum):
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    ENCODING = "encoding"
    # Add more preprocessing options as needed

class Modes(str, Enum):
    TRAIN = "train"
    TEST = "test"
    # Add more modes as needed

class Models(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    # Add more models as needed
