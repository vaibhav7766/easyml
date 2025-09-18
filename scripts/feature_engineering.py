import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import PolynomialFeatures
import joblib
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['feature_engineering']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(data_path):
    """Load prepared data"""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def scale_features(X, method='standard', scaler_path=None):
    """Scale numerical features"""
    logger.info(f"Scaling features using {method} method")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Save scaler
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    return X_scaled, scaler

def encode_categorical_features(X, y=None, strategy='target', encoder_path=None):
    """Encode categorical features"""
    logger.info(f"Encoding categorical features using {strategy} strategy")
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        logger.info("No categorical columns found")
        return X, None
    
    X_encoded = X.copy()
    encoders = {}
    
    for col in categorical_cols:
        if strategy == 'onehot':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(X[[col]])
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
            X_encoded = X_encoded.drop(col, axis=1)
            X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
            
        elif strategy == 'target' and y is not None:
            encoder = TargetEncoder()
            X_encoded[col] = encoder.fit_transform(X[[col]], y)
            
        elif strategy == 'label':
            encoder = LabelEncoder()
            X_encoded[col] = encoder.fit_transform(X[col])
        
        encoders[col] = encoder
    
    # Save encoders
    if encoder_path:
        joblib.dump(encoders, encoder_path)
        logger.info(f"Encoders saved to {encoder_path}")
    
    return X_encoded, encoders

def select_features(X, y, method='mutual_info', max_features=50):
    """Select best features"""
    logger.info(f"Selecting features using {method} method, max features: {max_features}")
    
    if method == 'mutual_info':
        if y.dtype == 'object' or len(np.unique(y)) < 10:
            # Classification
            score_func = mutual_info_classif
        else:
            # Regression
            from sklearn.feature_selection import mutual_info_regression
            score_func = mutual_info_regression
    elif method == 'chi2':
        score_func = chi2
    elif method == 'f_score':
        if y.dtype == 'object' or len(np.unique(y)) < 10:
            score_func = f_classif
        else:
            from sklearn.feature_selection import f_regression
            score_func = f_regression
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Select features
    k = min(max_features, X.shape[1])
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    logger.info(f"Selected {len(selected_features)} features")
    return X_selected, selector, selected_features

def create_polynomial_features(X, degree=2, max_features=1000):
    """Create polynomial features"""
    logger.info(f"Creating polynomial features with degree {degree}")
    
    # Limit to numeric columns only and first few columns to avoid explosion
    numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 numeric columns
    X_poly_input = X[numeric_cols]
    
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_poly_input)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(X_poly_input.columns)
    
    # Limit number of features
    if len(feature_names) > max_features:
        X_poly = X_poly[:, :max_features]
        feature_names = feature_names[:max_features]
    
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    # Combine with original features
    X_combined = pd.concat([X.drop(numeric_cols, axis=1), X_poly_df], axis=1)
    
    logger.info(f"Created {len(feature_names)} polynomial features")
    return X_combined, poly

def create_interaction_features(X, max_interactions=50):
    """Create interaction features"""
    logger.info(f"Creating interaction features, max: {max_interactions}")
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        logger.info("Not enough numeric columns for interactions")
        return X
    
    X_interactions = X.copy()
    interaction_count = 0
    
    for i, col1 in enumerate(numeric_cols[:10]):  # Limit to avoid explosion
        for col2 in numeric_cols[i+1:10]:
            if interaction_count >= max_interactions:
                break
            
            interaction_name = f"{col1}_x_{col2}"
            X_interactions[interaction_name] = X[col1] * X[col2]
            interaction_count += 1
        
        if interaction_count >= max_interactions:
            break
    
    logger.info(f"Created {interaction_count} interaction features")
    return X_interactions

def engineer_features(input_path, output_dir, target_column=None):
    """Main feature engineering function"""
    logger.info("Starting feature engineering...")
    
    # Load parameters
    params = load_parameters()
    
    # Load data
    data = load_data(input_path)
    
    # Separate features and target
    if target_column and target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        target_column = data.columns[-1]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode categorical features
    encoder_path = os.path.join(output_dir, 'encoders.joblib')
    X_encoded, encoders = encode_categorical_features(
        X, y, 
        strategy=params.get('encoding_strategy', 'target'),
        encoder_path=encoder_path
    )
    
    # Create polynomial features
    if params.get('create_polynomial_features', False):
        poly_path = os.path.join(output_dir, 'polynomial_transformer.joblib')
        X_encoded, poly_transformer = create_polynomial_features(
            X_encoded, 
            degree=params.get('polynomial_degree', 2)
        )
        joblib.dump(poly_transformer, poly_path)
    
    # Create interaction features
    if params.get('create_interaction_features', False):
        X_encoded = create_interaction_features(X_encoded)
    
    # Scale features
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    X_scaled, scaler = scale_features(
        X_encoded, 
        method=params.get('scaling_method', 'standard'),
        scaler_path=scaler_path
    )
    
    # Feature selection
    if params.get('feature_selection', False):
        selector_path = os.path.join(output_dir, 'feature_selector.joblib')
        X_selected, selector, selected_features = select_features(
            X_scaled, y,
            method=params.get('feature_selection_method', 'mutual_info'),
            max_features=params.get('max_features', 50)
        )
        joblib.dump(selector, selector_path)
        
        # Save selected features list
        features_path = os.path.join(output_dir, 'selected_features.json')
        with open(features_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
    else:
        X_selected = X_scaled
        selected_features = list(X_scaled.columns)
    
    # Combine final features with target
    final_data = pd.concat([X_selected, y], axis=1)
    
    # Save engineered data
    output_path = os.path.join(output_dir, 'engineered_data.csv')
    final_data.to_csv(output_path, index=False)
    
    # Save feature engineering info
    info_path = os.path.join(output_dir, 'feature_engineering_info.json')
    feature_info = {
        'original_features': list(X.columns),
        'final_features': selected_features,
        'target_column': target_column,
        'original_shape': X.shape,
        'final_shape': X_selected.shape,
        'encoding_strategy': params.get('encoding_strategy', 'target'),
        'scaling_method': params.get('scaling_method', 'standard'),
        'feature_selection': params.get('feature_selection', False)
    }
    
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info(f"Feature engineering completed. Output saved to {output_dir}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline')
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--target', help='Target column name')
    
    args = parser.parse_args()
    
    try:
        engineer_features(args.input, args.output, args.target)
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        sys.exit(1)
