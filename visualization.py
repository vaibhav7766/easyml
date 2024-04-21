import pandas as pd
import plotly.express as px


class Visualization:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def plot(self, x: str | None = None, y: str | None = None) -> str:
        fig = px.line(self.df, x=x, y=y)
        return fig.to_html(full_html=False)

    def bar(self, x: str | None = None, y: str | None = None) -> str:
        fig = px.bar(self.df, x=x, y=y)
        return fig.to_html(full_html=False)

    def scatter(self, x: str | None = None, y: str | None = None) -> str:
        fig = px.scatter(self.df, x=x, y=y)
        return fig.to_html(full_html=False)

    def boxplot(self, x: str | None = None, y: str | None = None) -> str:
        fig = px.box(self.df, x=x, y=y)
        return fig.to_html(full_html=False)

    def hist(self, x: str | None = None, y: str | None = None) -> str:
        fig = px.histogram(self.df, x=x, y=y)
        return fig.to_html(full_html=False)

    def heatmap(self, corr_matrix: pd.DataFrame | None = None) -> str:
        if corr_matrix is None:
            fig = px.imshow(self.df, text_auto=True)
        else:
            fig = px.imshow(corr_matrix, text_auto=True)
        return fig.to_html(full_html=False)
