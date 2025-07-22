"""
Visualization service for handling all plot generation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from scipy.stats import chi2_contingency
from typing import Optional, Any

from app.core.config import settings
from app.core.enums import PlotType


class VisualizationService:
    """Service for generating data visualizations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.figure_size = settings.default_figure_size
    
    def _to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    
    def generate_plot(
        self, 
        plot_type: PlotType, 
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        target: Optional[str] = None,
        model: Optional[Any] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate plot based on type and parameters"""
        
        plt.figure(figsize=self.figure_size)
        
        # Numeric Feature Visualizations
        if plot_type == PlotType.HISTOGRAM:
            return self._create_histogram(x)
        elif plot_type == PlotType.SCATTER:
            return self._create_scatter(x, y, hue)
        elif plot_type == PlotType.CORRELATION_MATRIX:
            return self._create_correlation_matrix()
        elif plot_type == PlotType.BOXPLOT:
            return self._create_boxplot(x, y)
        elif plot_type == PlotType.PAIRPLOT:
            return self._create_pairplot(hue, top_k)
        elif plot_type == PlotType.VIOLIN:
            return self._create_violin(x, y)
        elif plot_type == PlotType.KDE:
            return self._create_kde(x)
        
        # Categorical Feature Visualizations
        elif plot_type == PlotType.COUNTPLOT:
            return self._create_countplot(x)
        elif plot_type == PlotType.PIE:
            return self._create_pie(x)
        elif plot_type == PlotType.TARGET_MEAN:
            return self._create_target_mean(x, target)
        elif plot_type == PlotType.STACKED_BAR:
            return self._create_stacked_bar(x, y)
        elif plot_type == PlotType.CHI_SQUARED_HEATMAP:
            return self._create_chi_squared_heatmap()
        
        # Mixed & ML Use-Case Visualizations
        elif plot_type == PlotType.PCA_SCATTER:
            return self._create_pca_scatter(hue)
        elif plot_type == PlotType.CLASS_IMBALANCE:
            return self._create_class_imbalance(target)
        elif plot_type == PlotType.LEARNING_CURVE:
            return self._create_learning_curve(target, model)
        
        else:
            raise ValueError(f"Plot type '{plot_type}' is not supported.")
    
    def _create_histogram(self, x: str) -> str:
        """Create histogram plot"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid column for histogram.")
        
        sns.histplot(self.data[x], kde=True)
        plt.title(f"Distribution of {x}")
        return self._to_base64()
    
    def _create_scatter(self, x: str, y: str, hue: Optional[str]) -> str:
        """Create scatter plot"""
        if not x or not y or x not in self.data.columns or y not in self.data.columns:
            raise ValueError("Please specify valid x and y columns for scatter plot.")
        
        hue_data = self.data[hue] if hue and hue in self.data.columns else None
        sns.scatterplot(x=self.data[x], y=self.data[y], hue=hue_data)
        plt.title(f"Scatter Plot: {x} vs {y}")
        return self._to_base64()
    
    def _create_correlation_matrix(self) -> str:
        """Create correlation matrix heatmap"""
        corr = self.data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Correlation Matrix")
        return self._to_base64()
    
    def _create_boxplot(self, x: str, y: Optional[str] = None) -> str:
        """Create box plot"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid column for boxplot.")
        
        if y and y in self.data.columns:  # Grouped boxplot
            sns.boxplot(x=self.data[y], y=self.data[x])
            plt.title(f"Box Plot: {x} by {y}")
        else:  # Single variable boxplot
            sns.boxplot(x=self.data[x])
            plt.title(f"Box Plot of {x}")
        
        return self._to_base64()
    
    def _create_pairplot(self, hue: Optional[str], top_k: Optional[int]) -> str:
        """Create pairplot"""
        numeric_cols = self.data.select_dtypes(include='number').columns
        
        if top_k:
            numeric_cols = numeric_cols[:top_k]
        
        subset_data = self.data[numeric_cols]
        if hue and hue in self.data.columns:
            subset_data = subset_data.copy()
            subset_data[hue] = self.data[hue]
        
        sns.pairplot(subset_data, hue=hue if hue else None)
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        return self._to_base64()
    
    def _create_violin(self, x: str, y: Optional[str] = None) -> str:
        """Create violin plot"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid column for violin plot.")
        
        if y and y in self.data.columns:  # Grouped violin plot
            sns.violinplot(x=self.data[y], y=self.data[x])
            plt.title(f"Violin Plot: {x} by {y}")
        else:  # Single variable violin plot
            sns.violinplot(x=self.data[x])
            plt.title(f"Violin Plot of {x}")
        
        return self._to_base64()
    
    def _create_kde(self, x: str) -> str:
        """Create KDE plot"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid column for KDE plot.")
        
        sns.kdeplot(data=self.data, x=x, fill=True)
        plt.title(f"KDE Plot of {x}")
        return self._to_base64()
    
    def _create_countplot(self, x: str) -> str:
        """Create count plot"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid categorical column for countplot.")
        
        sns.countplot(data=self.data, x=x)
        plt.title(f"Count Plot of {x}")
        plt.xticks(rotation=45)
        return self._to_base64()
    
    def _create_pie(self, x: str) -> str:
        """Create pie chart"""
        if not x or x not in self.data.columns:
            raise ValueError("Please specify a valid categorical column for pie chart.")
        
        value_counts = self.data[x].value_counts()
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f"Pie Chart of {x}")
        return self._to_base64()
    
    def _create_target_mean(self, x: str, target: str) -> str:
        """Create target mean plot"""
        if not x or not target or x not in self.data.columns or target not in self.data.columns:
            raise ValueError("Please specify valid categorical column and target for target mean plot.")
        
        mean_values = self.data.groupby(x)[target].mean().sort_values()
        sns.barplot(x=mean_values.index, y=mean_values.values)
        plt.title(f"Target Mean Plot: {target} by {x}")
        plt.xticks(rotation=45)
        return self._to_base64()
    
    def _create_stacked_bar(self, x: str, y: str) -> str:
        """Create stacked bar chart"""
        if not x or not y or x not in self.data.columns or y not in self.data.columns:
            raise ValueError("Please specify valid categorical columns for stacked bar chart.")
        
        crosstab = pd.crosstab(self.data[x], self.data[y])
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f"Stacked Bar Chart: {x} by {y}")
        plt.xticks(rotation=45)
        plt.legend(title=y)
        return self._to_base64()
    
    def _create_chi_squared_heatmap(self) -> str:
        """Create chi-squared heatmap"""
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) < 2:
            raise ValueError("Need at least 2 categorical columns for chi-squared heatmap.")
        
        chi2_matrix = np.zeros((len(cat_cols), len(cat_cols)))
        for i, col1 in enumerate(cat_cols):
            for j, col2 in enumerate(cat_cols):
                if i != j:
                    try:
                        crosstab = pd.crosstab(self.data[col1], self.data[col2])
                        chi2_stat, _, _, _ = chi2_contingency(crosstab)
                        chi2_matrix[i][j] = chi2_stat
                    except:
                        chi2_matrix[i][j] = 0
                else:
                    chi2_matrix[i][j] = 0
        
        sns.heatmap(chi2_matrix, annot=True, fmt='.2f',
                   xticklabels=cat_cols, yticklabels=cat_cols,
                   cmap='viridis')
        plt.title("Chi-squared Statistics Heatmap")
        return self._to_base64()
    
    def _create_pca_scatter(self, hue: Optional[str]) -> str:
        """Create PCA scatter plot"""
        numeric_data = self.data.select_dtypes(include='number')
        if numeric_data.shape[1] < 2:
            raise ValueError("Need at least 2 numeric features for PCA.")
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data.fillna(0))
        
        color_data = None
        if hue and hue in self.data.columns:
            color_data = self.data[hue].astype('category').cat.codes
        
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=color_data, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title("PCA Scatter Plot (2D)")
        
        if color_data is not None:
            plt.colorbar(label=hue)
        
        return self._to_base64()
    
    def _create_class_imbalance(self, target: str) -> str:
        """Create class imbalance plot"""
        if not target or target not in self.data.columns:
            raise ValueError("Please specify a valid target column for class imbalance plot.")
        
        class_counts = self.data[target].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title(f"Class Distribution - {target}")
        plt.xlabel("Classes")
        plt.ylabel("Count")
        
        # Add percentage labels
        total = len(self.data)
        for i, count in enumerate(class_counts.values):
            plt.text(i, count + total*0.01, f'{count/total:.1%}',
                    ha='center', va='bottom')
        
        return self._to_base64()
    
    def _create_learning_curve(self, target: str, model: Any) -> str:
        """Create learning curve plot"""
        if model is None or not target:
            raise ValueError("Model and target are required for learning curve.")
        
        numeric_data = self.data.select_dtypes(include='number')
        X = numeric_data.drop(columns=[target] if target in numeric_data.columns else [])
        y = self.data[target]
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X.fillna(0), y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy' if y.dtype == 'object' else 'r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        return self._to_base64()
