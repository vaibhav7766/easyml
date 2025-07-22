"""
Test script for EasyML Visualization System
Demonstrates all the visualization types implemented from Notion database
"""

import requests
import base64
import io
from PIL import Image
import pandas as pd
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust based on your server
USER_ID = "test_user"
PROJECT_ID = "test_project"

def save_image_from_base64(base64_string, filename):
    """Save base64 encoded image to file"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(filename)
    print(f"Saved visualization: {filename}")

def test_visualization(plot_type, **params):
    """Test a specific visualization type"""
    url = f"{BASE_URL}/visualizations"
    
    # Base parameters
    query_params = {
        "project_id": PROJECT_ID,
        "user_id": USER_ID,
        "plot_type": plot_type
    }
    
    # Add additional parameters
    query_params.update({k: v for k, v in params.items() if v is not None})
    
    try:
        response = requests.get(url, params=query_params)
        
        if response.status_code == 200:
            data = response.json()
            if "image" in data:
                filename = f"{plot_type}_{params.get('x', 'default')}.png"
                save_image_from_base64(data["image"], filename)
                return True
            else:
                print(f"Error for {plot_type}: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"HTTP Error {response.status_code} for {plot_type}")
            return False
            
    except Exception as e:
        print(f"Exception testing {plot_type}: {str(e)}")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'score': np.random.normal(75, 15, n_samples),
        'target_binary': np.random.choice([0, 1], n_samples),
        'target_multi': np.random.choice(['Class1', 'Class2', 'Class3'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create uploads directory
    import os
    os.makedirs('uploads', exist_ok=True)
    
    # Save the dataset
    df.to_csv(f'uploads/{USER_ID}_{PROJECT_ID}.csv', index=False)
    print(f"Created sample dataset: uploads/{USER_ID}_{PROJECT_ID}.csv")
    
    return df

def run_all_tests():
    """Run comprehensive tests for all visualization types"""
    
    print("=== EasyML Visualization System Tests ===")
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Test results tracking
    results = {}
    
    print("=== Testing Numeric Feature Visualizations ===")
    
    # Histogram
    results['histogram'] = test_visualization('histogram', x='age')
    
    # Scatter plot
    results['scatter'] = test_visualization('scatter', x='age', y='income', hue='category')
    
    # Correlation matrix
    results['correlation_matrix'] = test_visualization('correlation_matrix')
    
    # Box plot (single variable)
    results['boxplot_single'] = test_visualization('boxplot', x='income')
    
    # Box plot (grouped)
    results['boxplot_grouped'] = test_visualization('boxplot', x='income', y='category')
    
    # Pairplot
    results['pairplot'] = test_visualization('pairplot', hue='target_multi', top_k=4)
    
    # Violin plot (single variable)
    results['violin_single'] = test_visualization('violin', x='score')
    
    # Violin plot (grouped)
    results['violin_grouped'] = test_visualization('violin', x='score', y='category')
    
    # KDE plot
    results['kde'] = test_visualization('kde', x='age')
    
    print()
    print("=== Testing Categorical Feature Visualizations ===")
    
    # Count plot
    results['countplot'] = test_visualization('countplot', x='category')
    
    # Pie chart
    results['pie'] = test_visualization('pie', x='region')
    
    # Target mean plot
    results['target_mean'] = test_visualization('target_mean', x='category', target='score')
    
    # Stacked bar chart
    results['stacked_bar'] = test_visualization('stacked_bar', x='category', y='region')
    
    # Chi-squared heatmap
    results['chi_squared_heatmap'] = test_visualization('chi_squared_heatmap')
    
    print()
    print("=== Testing Mixed & ML Use-Case Visualizations ===")
    
    # PCA scatter plot
    results['pca_scatter'] = test_visualization('pca_scatter', hue='target_multi')
    
    # Class imbalance plot
    results['class_imbalance'] = test_visualization('class_imbalance', target='target_multi')
    
    # Learning curve (might fail without trained model)
    results['learning_curve'] = test_visualization('learning_curve', target='target_binary')
    
    print()
    print("=== Test Results Summary ===")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    print()
    
    print("Detailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if successful == total:
        print("\n🎉 All visualization tests passed!")
    else:
        print(f"\n⚠️  {total - successful} tests failed. Check implementation.")

if __name__ == "__main__":
    run_all_tests()
