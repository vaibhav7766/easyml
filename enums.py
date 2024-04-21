from enum import Enum


class Plots(str, Enum):
    plot = "plot"
    bar = "bar"
    scatter = "scatter"
    boxplot = "boxplot"
    hist = "hist"
    heatmap = "heatmap"
