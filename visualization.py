import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

class Visualization:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _to_base64(self):
        """Convert current matplotlib plot to base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64

    def plot(self, plot_type: str, x=None, y=None, hue=None):
        """Generate different kinds of plots."""
        if plot_type == "histogram":
            if x and x in self.data.columns:
                sns.histplot(self.data[x], kde=True)
            else:
                raise ValueError("Please specify a valid column for histogram.")
        
        elif plot_type == "scatter":
            if x and y and x in self.data.columns and y in self.data.columns:
                sns.scatterplot(x=self.data[x], y=self.data[y], hue=self.data[hue] if hue else None)
            else:
                raise ValueError("Please specify valid x and y columns for scatter plot.")
        
        elif plot_type == "correlation_matrix":
            corr = self.data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        
        elif plot_type == "boxplot":
            if x and x in self.data.columns:
                sns.boxplot(x=self.data[x])
            else:
                raise ValueError("Please specify a valid column for boxplot.")

        elif plot_type == "pairplot":
            sns.pairplot(self.data.select_dtypes(include='number'))

        elif plot_type == "countplot":
            if x and x in self.data.columns:
                sns.countplot(x=self.data[x])
            else:
                raise ValueError("Please specify a valid categorical column for countplot.")

        else:
            raise ValueError(f"Plot type '{plot_type}' is not supported.")

        return self._to_base64()
