# EasyML Visualization Guide

This guide documents all the visualization features implemented in EasyML based on your Notion database requirements.

## Overview

The visualization system supports three main categories of plots:
1. **Numeric Feature Visualizations** - For analyzing numerical data
2. **Categorical Feature Visualizations** - For analyzing categorical data  
3. **Mixed & ML Use-Case Visualizations** - For advanced analysis and ML insights

## API Usage

### Endpoint
```
GET /visualizations
```

### Parameters
- `project_id` (required): Project identifier
- `user_id` (required): User identifier
- `plot_type` (required): Type of visualization to generate
- `x` (optional): Column name for x-axis
- `y` (optional): Column name for y-axis
- `hue` (optional): Column name for color grouping
- `target` (optional): Target column name (for supervised learning plots)
- `top_k` (optional): Number of top features to include (for pairplot)

### Example Request
```
GET /visualizations?project_id=123&user_id=user1&plot_type=histogram&x=age
```

## Supported Visualizations

### 1. Numeric Feature Visualizations

#### 📊 Histogram (`histogram`)
**Purpose**: Show distribution of numeric data  
**Parameters**: `x` (required)  
**Use Case**: Understanding data distribution, detecting skewness, outliers  
```
GET /visualizations?plot_type=histogram&x=column_name
```

#### 🔍 Scatter Plot (`scatter`)
**Purpose**: Show relationship between two numeric variables  
**Parameters**: `x` (required), `y` (required), `hue` (optional)  
**Use Case**: Linear/nonlinear patterns, correlation analysis  
```
GET /visualizations?plot_type=scatter&x=feature1&y=feature2&hue=category
```

#### 🔁 Correlation Heatmap (`correlation_matrix`)
**Purpose**: Show correlation between all numeric features  
**Parameters**: None required  
**Use Case**: Detect multicollinearity, feature redundancy  
```
GET /visualizations?plot_type=correlation_matrix
```

#### 🧊 Box Plot (`boxplot`)
**Purpose**: Show distribution, outliers, and quartiles  
**Parameters**: `x` (required), `y` (optional for grouped boxplot)  
**Use Case**: Outlier detection, comparing distributions across groups  
```
GET /visualizations?plot_type=boxplot&x=numeric_column
GET /visualizations?plot_type=boxplot&x=numeric_column&y=category_column
```

#### 🧮 Pairplot (`pairplot`)
**Purpose**: Show pairwise relationships between numeric features  
**Parameters**: `hue` (optional), `top_k` (optional)  
**Use Case**: Comprehensive feature relationship analysis  
```
GET /visualizations?plot_type=pairplot&hue=target&top_k=5
```

#### 🎻 Violin Plot (`violin`)
**Purpose**: Show distribution shape and density  
**Parameters**: `x` (required), `y` (optional for grouped violin)  
**Use Case**: Distribution shape analysis, comparing densities  
```
GET /visualizations?plot_type=violin&x=numeric_column
GET /visualizations?plot_type=violin&x=numeric_column&y=category_column
```

#### 🌊 KDE Plot (`kde`)
**Purpose**: Show smooth density estimation  
**Parameters**: `x` (required)  
**Use Case**: Smooth density visualization, distribution analysis  
```
GET /visualizations?plot_type=kde&x=numeric_column
```

### 2. Categorical Feature Visualizations

#### 📊 Count Plot (`countplot`)
**Purpose**: Show frequency of categorical values  
**Parameters**: `x` (required)  
**Use Case**: Category frequency analysis  
```
GET /visualizations?plot_type=countplot&x=category_column
```

#### 🧩 Pie Chart (`pie`)
**Purpose**: Show proportions when few categories exist  
**Parameters**: `x` (required)  
**Use Case**: Proportion visualization for limited categories  
```
GET /visualizations?plot_type=pie&x=category_column
```

#### 📦 Box Plot - Numeric vs Category (`boxplot`)
**Purpose**: Show numeric distribution across categorical groups  
**Parameters**: `x` (numeric), `y` (categorical)  
**Use Case**: Compare numeric spreads across categories  
```
GET /visualizations?plot_type=boxplot&x=numeric_column&y=category_column
```

#### 🎯 Target Mean Plot (`target_mean`)
**Purpose**: Show average target value for each category  
**Parameters**: `x` (categorical), `target` (required)  
**Use Case**: Understand category impact on target variable  
```
GET /visualizations?plot_type=target_mean&x=category_column&target=target_column
```

#### 📊 Stacked Bar Chart (`stacked_bar`)
**Purpose**: Show cross-category distribution  
**Parameters**: `x` (categorical), `y` (categorical)  
**Use Case**: Analyze relationship between two categorical variables  
```
GET /visualizations?plot_type=stacked_bar&x=category1&y=category2
```

#### 📉 Chi-squared Heatmap (`chi_squared_heatmap`)
**Purpose**: Show independence between categorical variables  
**Parameters**: None (uses all categorical columns)  
**Use Case**: Detect relationships between categorical features  
```
GET /visualizations?plot_type=chi_squared_heatmap
```

#### 🎻 Violin Plot - Numeric vs Category (`violin`)
**Purpose**: Show distribution across categorical groups  
**Parameters**: `x` (numeric), `y` (categorical)  
**Use Case**: Compare distributions across categories  
```
GET /visualizations?plot_type=violin&x=numeric_column&y=category_column
```

### 3. Mixed & ML Use-Case Visualizations

#### 🌐 PCA Scatter Plot (`pca_scatter`)
**Purpose**: Dimensionality reduction visualization  
**Parameters**: `hue` (optional for coloring)  
**Use Case**: Data structure analysis, clustering visualization  
```
GET /visualizations?plot_type=pca_scatter&hue=target_column
```

#### 📊 Class Imbalance Plot (`class_imbalance`)
**Purpose**: Show target class distribution  
**Parameters**: `target` (required)  
**Use Case**: Detect class imbalance for classification tasks  
```
GET /visualizations?plot_type=class_imbalance&target=target_column
```

#### 📈 Learning Curve (`learning_curve`)
**Purpose**: Show model performance vs training size  
**Parameters**: `target` (required), requires trained model  
**Use Case**: Diagnose overfitting/underfitting  
**Note**: Currently requires model integration  
```
GET /visualizations?plot_type=learning_curve&target=target_column
```

## Implementation Status

Based on your Notion database, all "In Progress" visualizations have been implemented:

### ✅ Completed from Notion "In Progress" List:

**Numeric Features:**
- ✅ 📊 Histogram
- ✅ 🧮 Pairplot (top-k features)
- ✅ 🔁 Correlation Heatmap
- ✅ 🎻 Violin Plot
- ✅ 🌊 KDE Plot
- ✅ 🧊 Box Plot
- ✅ 🔍 Scatter Plot (vs target)

**Categorical Features:**
- ✅ 📊 Count Plot
- ✅ 🧩 Pie Chart
- ✅ 📦 Box Plot (Numeric vs Category)
- ✅ 🎯 Target Mean Plot
- ✅ 📉 Chi-squared Heatmap
- ✅ 📊 Stacked Bar Chart
- ✅ 🎻 Violin Plot (Numeric vs Category)

**Mixed & ML Use-Cases:**
- ✅ 🌐 PCA Scatter Plot (2D)
- ✅ 📊 Class Imbalance Plot
- ✅ 📈 Learning Curve (optional)

## Error Handling

The API includes comprehensive error handling:
- Invalid column names
- Missing required parameters
- File not found errors
- Plotting errors with descriptive messages

## Response Format

```json
{
  "image": "base64_encoded_image_string",
  "plot_type": "histogram",
  "parameters": {
    "x": "column_name",
    "y": null,
    "hue": null,
    "target": null,
    "top_k": null
  }
}
```

Error response:
```json
{
  "error": "Error message describing the issue"
}
```

## Technical Notes

- All plots are returned as base64-encoded PNG images
- Automatic figure sizing (10x6 inches) for consistent output
- Proper handling of missing values and data type detection
- Color schemes optimized for data visualization
- Automatic rotation of axis labels when needed
- Statistical calculations for advanced plots (chi-squared, PCA)

## Dependencies

The visualization system requires:
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn (for PCA and learning curves)
- scipy (for chi-squared statistics)
