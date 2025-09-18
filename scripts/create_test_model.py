#!/usr/bin/env python3
"""
Create a test model for deployment testing
"""
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import os
from datetime import datetime

def create_test_model():
    """Create a simple test model with synthetic data"""
    print("🤖 Creating test model for deployment...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train a simple Random Forest model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )
    
    print("📊 Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    print(f"📈 Model Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Create models directory structure
    model_dir = "models/test_deployment_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model using both pickle and joblib
    model_path_pkl = f"{model_dir}/model.pkl"
    model_path_joblib = f"{model_dir}/model.joblib"
    
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(model, f)
    
    joblib.dump(model, model_path_joblib)
    
    # Save model metadata
    metadata = {
        "model_id": "test_model_001",
        "model_name": "Test Random Forest",
        "version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "features": [f"feature_{i}" for i in range(X.shape[1])],
        "target": "binary_classification",
        "metrics": metrics,
        "model_type": "classification",
        "framework": "scikit-learn",
        "created_at": datetime.utcnow().isoformat(),
        "file_paths": {
            "model_pkl": model_path_pkl,
            "model_joblib": model_path_joblib,
            "metadata": f"{model_dir}/metadata.json"
        },
        "input_shape": list(X.shape),
        "output_classes": ["class_0", "class_1"],
        "training_config": {
            "n_estimators": 50,
            "max_depth": 10,
            "random_state": 42
        }
    }
    
    # Save metadata
    with open(f"{model_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save sample data for testing
    sample_data = {
        "sample_input": X_test[:5].tolist(),
        "expected_output": y_test[:5].tolist(),
        "feature_names": [f"feature_{i}" for i in range(X.shape[1])]
    }
    
    with open(f"{model_dir}/sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"💾 Model saved to: {model_dir}")
    print(f"📋 Metadata saved with ID: {metadata['model_id']}")
    
    return metadata

if __name__ == "__main__":
    model_info = create_test_model()
    print("\n🎯 Test model created successfully!")
    print("📁 Files created:")
    for key, path in model_info["file_paths"].items():
        print(f"   {key}: {path}")
