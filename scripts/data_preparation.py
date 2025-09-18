import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import dvc.api
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['data_preparation']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(data_path):
    """Load data from the specified path"""
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def handle_missing_values(data, strategy):
    """Handle missing values in the dataset"""
    logger.info(f"Handling missing values with strategy: {strategy}")
    
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        if strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='constant', fill_value=0)
        
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    return data

def remove_outliers(data, method='iqr', threshold=1.5):
    """Remove outliers from numeric columns"""
    logger.info(f"Removing outliers using {method} method with threshold {threshold}")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    logger.info(f"Data shape after outlier removal: {data.shape}")
    return data

def prepare_data(input_path, output_dir):
    """Main data preparation function"""
    logger.info("Starting data preparation...")
    
    # Load parameters
    params = load_parameters()
    
    # Load data
    data = load_data(input_path)
    
    # Remove duplicates
    if params.get('remove_duplicates', True):
        initial_shape = data.shape
        data = data.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows")
    
    # Handle missing values
    if params.get('handle_missing_values'):
        data = handle_missing_values(data, params['handle_missing_values'])
    
    # Remove outliers
    if params.get('outlier_detection'):
        data = remove_outliers(
            data, 
            method=params['outlier_detection'], 
            threshold=params.get('outlier_threshold', 1.5)
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    output_path = os.path.join(output_dir, 'prepared_data.csv')
    data.to_csv(output_path, index=False)
    
    # Save data info
    info_path = os.path.join(output_dir, 'data_info.json')
    data_info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    
    import json
    with open(info_path, 'w') as f:
        json.dump(data_info, f, indent=2, default=str)
    
    logger.info(f"Data preparation completed. Output saved to {output_dir}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preparation Pipeline')
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        prepare_data(args.input, args.output)
        logger.info("Data preparation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Data preparation pipeline failed: {e}")
        sys.exit(1)
