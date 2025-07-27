"""
Visualization service for handling all plot generation using Plotly
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from scipy.stats import chi2_contingency
from typing import Optional, Any, Dict, List

from app.core.config import get_settings
from app.core.enums import PlotType


class VisualizationService:
    """Service for generating interactive data visualizations using Plotly"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.settings = get_settings()
        # Default theme and styling
        self.theme = "plotly_white"
        self.color_palette = px.colors.qualitative.Set1
    
    def create_plot(
        self, 
        plot_type: PlotType, 
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate interactive plot using Plotly
        
        Returns:
            Dictionary containing plotly JSON, plot info, and metadata
        """
        try:
            # Ensure plot_type is a PlotType enum
            if isinstance(plot_type, str):
                plot_type = PlotType(plot_type)
            
            # Generate the plot based on type
            fig = self._generate_plot(plot_type, x_column, y_column, color_column, **kwargs)
            
            # Convert to JSON for frontend
            plotly_json = fig.to_json()
            
            return {
                "plotly_json": plotly_json,
                "plot_type": plot_type.value if hasattr(plot_type, 'value') else str(plot_type),
                "plot_info": {
                    "columns_used": {
                        "x_column": x_column,
                        "y_column": y_column, 
                        "color_column": color_column
                    },
                    "data_shape": self.data.shape,
                    "plot_library": "plotly",
                    "interactive": True
                }
            }
            
        except Exception as e:
            plot_type_name = plot_type.value if hasattr(plot_type, 'value') else str(plot_type)
            return {"error": f"Error generating {plot_type_name} plot: {str(e)}"}
    
    def _generate_plot(
        self,
        plot_type: PlotType,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Generate the appropriate plot based on type"""
        
        # Route to appropriate plot method
        plot_methods = {
            PlotType.HISTOGRAM: self._create_histogram,
            PlotType.SCATTER: self._create_scatter,
            PlotType.CORRELATION_MATRIX: self._create_correlation_matrix,
            PlotType.BOXPLOT: self._create_boxplot,
            PlotType.PAIRPLOT: self._create_pairplot,
            PlotType.VIOLIN: self._create_violin,
            PlotType.KDE: self._create_kde,
            PlotType.COUNTPLOT: self._create_countplot,
            PlotType.PIE: self._create_pie,
            PlotType.TARGET_MEAN: self._create_target_mean,
            PlotType.STACKED_BAR: self._create_stacked_bar,
            PlotType.CHI_SQUARED_HEATMAP: self._create_chi_squared_heatmap,
            PlotType.PCA_SCATTER: self._create_pca_scatter,
            PlotType.CLASS_IMBALANCE: self._create_class_imbalance,
            PlotType.LEARNING_CURVE: self._create_learning_curve
        }
        
        if plot_type not in plot_methods:
            raise ValueError(f"Plot type '{plot_type}' is not supported.")
        
        return plot_methods[plot_type](x_column, y_column, color_column, **kwargs)
    
    def _create_histogram(self, column: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive histogram for numeric column"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        fig = px.histogram(
            self.data, 
            x=column,
            nbins=kwargs.get('bins', 30),
            title=f'Histogram of {column}',
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
    
    def _create_scatter(self, x: str, y: str, color_column: Optional[str] = None, **kwargs) -> go.Figure:
        """Create interactive scatter plot"""
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError("Invalid column(s)")
        
        fig = px.scatter(
            self.data,
            x=x,
            y=y,
            color=color_column if color_column and color_column in self.data.columns else None,
            title=f'Scatter Plot: {x} vs {y}',
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y
        )
        
        return fig
    
    def _create_correlation_matrix(self, x_column=None, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive correlation heatmap for numeric columns"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title='Correlation Matrix',
            template=self.theme,
            color_continuous_scale='RdBu'
        )
        
        return fig
    
    def _create_boxplot(self, x: str, y: Optional[str] = None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive box plot"""
        if x not in self.data.columns:
            raise ValueError(f"Column '{x}' not found")
        
        if y and y in self.data.columns:
            fig = px.box(
                self.data,
                x=x,
                y=y,
                title=f'Box Plot: {y} by {x}',
                template=self.theme
            )
        else:
            fig = px.box(
                self.data,
                y=x,
                title=f'Box Plot of {x}',
                template=self.theme
            )
        
        return fig
    
    def _create_pairplot(self, x_column=None, y_column=None, color_column: Optional[str] = None, **kwargs) -> go.Figure:
        """Create interactive pairplot (scatter matrix) for numeric features"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Limit to top_k features if specified
        top_k = kwargs.get('top_k')
        if top_k and len(numeric_data.columns) > top_k:
            numeric_data = numeric_data.iloc[:, :top_k]
        
        plot_data = numeric_data.copy()
        if color_column and color_column in self.data.columns:
            plot_data[color_column] = self.data[color_column]
        
        fig = px.scatter_matrix(
            plot_data,
            dimensions=numeric_data.columns,
            color=color_column if color_column and color_column in plot_data.columns else None,
            title='Pairplot of Numeric Features',
            template=self.theme
        )
        
        return fig
    
    def _create_violin(self, x: str, y: Optional[str] = None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive violin plot"""
        if x not in self.data.columns:
            raise ValueError(f"Column '{x}' not found")
        
        if y and y in self.data.columns:
            fig = px.violin(
                self.data,
                x=x,
                y=y,
                title=f'Violin Plot: {y} by {x}',
                template=self.theme,
                box=True
            )
        else:
            fig = px.violin(
                self.data,
                y=x,
                title=f'Violin Plot of {x}',
                template=self.theme,
                box=True
            )
        
        return fig
    
    def _create_kde(self, column: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create KDE plot using Plotly"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Create KDE using distplot
        hist_data = [self.data[column].dropna()]
        group_labels = [column]
        
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            show_hist=False,
            show_rug=False
        )
        
        fig.update_layout(
            title=f'KDE Plot of {column}',
            xaxis_title=column,
            yaxis_title='Density',
            template=self.theme
        )
        
        return fig
    
    def _create_countplot(self, column: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive count plot for categorical data"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        value_counts = self.data[column].value_counts()
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Count Plot of {column}',
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Count'
        )
        
        return fig
    
    def _create_pie(self, column: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive pie chart"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        value_counts = self.data[column].value_counts()
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Pie Chart of {column}',
            template=self.theme
        )
        
        return fig
    
    def _create_target_mean(self, x_column: str, y_column: str, color_column=None, **kwargs) -> go.Figure:
        """Create target mean plot"""
        if x_column not in self.data.columns or y_column not in self.data.columns:
            raise ValueError("Invalid column(s)")
        
        target_means = self.data.groupby(x_column)[y_column].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=target_means.index,
            y=target_means.values,
            title=f'Mean {y_column} by {x_column}',
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=f'Mean {y_column}'
        )
        
        return fig
    
    def _create_stacked_bar(self, x: str, y: str, color_column=None, **kwargs) -> go.Figure:
        """Create interactive stacked bar chart"""
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError("Invalid column(s)")
        
        # Create crosstab
        cross_tab = pd.crosstab(self.data[x], self.data[y])
        
        fig = go.Figure()
        
        for col in cross_tab.columns:
            fig.add_trace(go.Bar(
                name=str(col),
                x=cross_tab.index,
                y=cross_tab[col],
                text=cross_tab[col],
                textposition='inside'
            ))
        
        fig.update_layout(
            title=f'Stacked Bar Chart: {x} vs {y}',
            xaxis_title=x,
            yaxis_title='Count',
            barmode='stack',
            template=self.theme
        )
        
        return fig
    
    def _create_chi_squared_heatmap(self, x_column=None, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create chi-squared test heatmap for categorical variables"""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) < 2:
            raise ValueError("Need at least 2 categorical columns for chi-squared analysis")
        
        chi2_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
        
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2:
                    contingency_table = pd.crosstab(self.data[col1], self.data[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    chi2_matrix.loc[col1, col2] = p_value
                else:
                    chi2_matrix.loc[col1, col2] = 1.0
        
        chi2_matrix = chi2_matrix.astype(float)
        
        fig = px.imshow(
            chi2_matrix,
            text_auto=True,
            aspect="auto",
            title='Chi-squared Test P-values Heatmap',
            template=self.theme,
            color_continuous_scale='RdBu'
        )
        
        return fig
    
    def _create_pca_scatter(self, x_column=None, y_column=None, color_column: Optional[str] = None, **kwargs) -> go.Figure:
        """Create interactive PCA scatter plot"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")
        
        # Remove any rows with NaN values
        numeric_data = numeric_data.dropna()
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data)
        
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        
        if color_column and color_column in self.data.columns:
            pca_df[color_column] = self.data[color_column].values[:len(pca_df)]
        
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=color_column if color_column and color_column in pca_df.columns else None,
            title=f'PCA Scatter Plot (Explained Variance: {sum(pca.explained_variance_ratio_):.2%})',
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
        )
        
        return fig
    
    def _create_class_imbalance(self, target_col: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive class imbalance visualization"""
        if target_col not in self.data.columns:
            raise ValueError(f"Column '{target_col}' not found")
        
        class_counts = self.data[target_col].value_counts()
        
        fig = px.bar(
            x=class_counts.values,
            y=class_counts.index,
            orientation='h',
            title=f'Class Imbalance: {target_col}',
            template=self.theme,
            text=class_counts.values
        )
        
        fig.update_layout(
            xaxis_title='Count',
            yaxis_title='Classes'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        return fig
    
    def _create_learning_curve(self, target_col: str, y_column=None, color_column=None, **kwargs) -> go.Figure:
        """Create interactive learning curve plot"""
        if target_col not in self.data.columns:
            raise ValueError(f"Column '{target_col}' not found")
        
        model = kwargs.get('model')
        if model is None:
            # Use a simple dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Prepare features and target
        feature_cols = [col for col in self.data.columns if col != target_col]
        X = self.data[feature_cols].select_dtypes(include=[np.number])
        y = self.data[target_col]
        
        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean + train_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean - train_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='rgb(0,100,80)')
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean + val_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean - val_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,65,54,0.2)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='rgb(255,65,54)')
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            template=self.theme
        )
        
        return fig
    
