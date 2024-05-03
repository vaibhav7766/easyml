from enum import Enum


class Plots(str, Enum):
    plot = "plot"
    bar = "bar"
    scatter = "scatter"
    boxplot = "boxplot"
    hist = "hist"
    heatmap = "heatmap"


class Options(str, Enum):
    remove_nulls = "remove_nulls"
    encode = "encode"
    feature_scaling = "feature_scaling"


class Modes(str, Enum):
    median = "median"
    mean = "mean"
    mode = "mode"
    bfill = "bfill"
    ffill = "ffill"
    pairwise = "pairwise"
    column = "column"
    row = "row"
    onehot = "onehot"
    label = "label"
    standard = "standard"
    minmax = "minmax"
    none = "none"


class Models(str, Enum):
    liner_regression = "liner_regression"
    logistic_regression = "logistic_regression"
    decision_tree = "decision_tree"
    k_nearest_neighbors = "k_nearest_neighbors"
    support_vector_machines = "support_vector_machines"
    polynomial_regression = "polynomial_regression"
    lasso_regression = "lasso_regression"
    ridge_regression = "ridge_regression"
