import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from scipy.stats import chi2_contingency

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

    def plot(self, plot_type: str, x=None, y=None, hue=None, target=None, model=None, top_k=None):
        """Generate different kinds of plots."""
        plt.figure(figsize=(10, 6))
        
        # Numeric Feature Visualizations
        if plot_type == "histogram":
            if x and x in self.data.columns:
                sns.histplot(self.data[x], kde=True)
                plt.title(f"Distribution of {x}")
            else:
                raise ValueError("Please specify a valid column for histogram.")
        
        elif plot_type == "scatter":
            if x and y and x in self.data.columns and y in self.data.columns:
                sns.scatterplot(x=self.data[x], y=self.data[y], hue=self.data[hue] if hue and hue in self.data.columns else None)
                plt.title(f"Scatter Plot: {x} vs {y}")
            else:
                raise ValueError("Please specify valid x and y columns for scatter plot.")
        
        elif plot_type == "correlation_matrix":
            corr = self.data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
            plt.title("Correlation Matrix")
        
        elif plot_type == "boxplot":
            if x and x in self.data.columns:
                if y and y in self.data.columns:  # Numeric vs Category
                    sns.boxplot(x=self.data[y], y=self.data[x])
                    plt.title(f"Box Plot: {x} by {y}")
                else:
                    sns.boxplot(x=self.data[x])
                    plt.title(f"Box Plot of {x}")
            else:
                raise ValueError("Please specify a valid column for boxplot.")

        elif plot_type == "pairplot":
            numeric_cols = self.data.select_dtypes(include='number').columns
            if top_k:
                numeric_cols = numeric_cols[:top_k]
            subset_data = self.data[numeric_cols]
            if hue and hue in self.data.columns:
                subset_data[hue] = self.data[hue]
            sns.pairplot(subset_data, hue=hue if hue else None)
            plt.suptitle("Pairplot of Numeric Features", y=1.02)

        elif plot_type == "countplot":
            if x and x in self.data.columns:
                sns.countplot(data=self.data, x=x)
                plt.title(f"Count Plot of {x}")
                plt.xticks(rotation=45)
            else:
                raise ValueError("Please specify a valid categorical column for countplot.")
                
        # New Numeric Visualizations
        elif plot_type == "violin":
            if x and x in self.data.columns:
                if y and y in self.data.columns:  # Numeric vs Category
                    sns.violinplot(x=self.data[y], y=self.data[x])
                    plt.title(f"Violin Plot: {x} by {y}")
                else:
                    sns.violinplot(x=self.data[x])
                    plt.title(f"Violin Plot of {x}")
            else:
                raise ValueError("Please specify a valid column for violin plot.")
                
        elif plot_type == "kde":
            if x and x in self.data.columns:
                sns.kdeplot(data=self.data, x=x, fill=True)
                plt.title(f"KDE Plot of {x}")
            else:
                raise ValueError("Please specify a valid column for KDE plot.")
        
        # Categorical Feature Visualizations
        elif plot_type == "pie":
            if x and x in self.data.columns:
                value_counts = self.data[x].value_counts()
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f"Pie Chart of {x}")
            else:
                raise ValueError("Please specify a valid categorical column for pie chart.")
                
        elif plot_type == "target_mean":
            if x and target and x in self.data.columns and target in self.data.columns:
                mean_values = self.data.groupby(x)[target].mean().sort_values()
                sns.barplot(x=mean_values.index, y=mean_values.values)
                plt.title(f"Target Mean Plot: {target} by {x}")
                plt.xticks(rotation=45)
            else:
                raise ValueError("Please specify valid categorical column and target for target mean plot.")
                
        elif plot_type == "stacked_bar":
            if x and y and x in self.data.columns and y in self.data.columns:
                crosstab = pd.crosstab(self.data[x], self.data[y])
                crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
                plt.title(f"Stacked Bar Chart: {x} by {y}")
                plt.xticks(rotation=45)
                plt.legend(title=y)
            else:
                raise ValueError("Please specify valid categorical columns for stacked bar chart.")
                
        elif plot_type == "chi_squared_heatmap":
            # Get only categorical columns
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) < 2:
                raise ValueError("Need at least 2 categorical columns for chi-squared heatmap.")
            
            # Calculate chi-squared statistics
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
            
        # Mixed & ML Use-Cases Visualizations
        elif plot_type == "pca_scatter":
            numeric_data = self.data.select_dtypes(include='number')
            if numeric_data.shape[1] < 2:
                raise ValueError("Need at least 2 numeric features for PCA.")
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_data.fillna(0))
            
            plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                       c=self.data[hue].astype('category').cat.codes if hue and hue in self.data.columns else 'blue',
                       alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title("PCA Scatter Plot (2D)")
            if hue and hue in self.data.columns:
                plt.colorbar(label=hue)
                
        elif plot_type == "class_imbalance":
            if target and target in self.data.columns:
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
            else:
                raise ValueError("Please specify a valid target column for class imbalance plot.")
                
        elif plot_type == "learning_curve":
            if model is None or target is None:
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

        else:
            raise ValueError(f"Plot type '{plot_type}' is not supported.")

        plt.tight_layout()
        return self._to_base64()
