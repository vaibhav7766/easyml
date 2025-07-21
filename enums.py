from enum import Enum

class Plots(str, Enum):
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    CORRELATION = "correlation"
    # Add more plot types as needed

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
