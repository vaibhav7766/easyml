"""
Comprehensive tests for training endpoints
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch
import tempfile
import os
import json

# Import the FastAPI app
from app.main import app
from app.core.enums import ModelType

# Create test client
client = TestClient(app)

# Test data - all numeric for ML model compatibility
SAMPLE_DATA = {
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "feature3": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Changed from strings to binary
    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

REGRESSION_DATA = {
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "target": [1.5, 2.8, 4.1, 5.6, 7.2, 8.8, 10.1, 11.7, 13.2, 14.9]
}

class TestTrainingEndpoints:
    """Test suite for training endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method to run before each test"""
        self.session_id = "test_session_123"
        self.file_id = None
        
    def upload_test_data(self, data_dict, filename="test_data.csv"):
        """Helper method to upload test data"""
        # Create a temporary CSV file
        df = pd.DataFrame(data_dict)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file_path = f.name
            
        try:
            # Upload the file
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/upload/",
                    files={"file": (filename, f, "text/csv")}
                )
            
            if response.status_code == 200:
                return response.json()["file_id"]
            else:
                pytest.fail(f"Failed to upload test data: {response.text}")
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_get_available_models(self):
        """Test the available models endpoint"""
        response = client.get("/api/v1/training/available-models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "models" in data
        assert "regression_models" in data
        assert "classification_models" in data
        
        # Check that we have all 21 models
        assert len(data["models"]) == 21
        
        # Check that models are properly categorized
        total_models = len(data["regression_models"]) + len(data["classification_models"])
        assert total_models == 21
        
        print(f"✅ Available models test passed - Found {len(data['models'])} models")
    
    def test_get_available_models_simple(self):
        """Test the simple available models endpoint"""
        response = client.get("/api/v1/training/models/available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "models" in data
        assert "classification" in data["models"]
        assert "regression" in data["models"]
        
        print("✅ Simple available models test passed")
    
    def test_get_model_hyperparameters(self):
        """Test getting default hyperparameters for a model"""
        model_type = "random_forest_classifier"
        response = client.get(f"/api/v1/training/models/{model_type}/hyperparameters")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_type"] == model_type
        assert "default_hyperparameters" in data
        
        print(f"✅ Model hyperparameters test passed for {model_type}")
    
    def test_train_classification_model(self):
        """Test training a classification model"""
        # Upload test data
        file_id = self.upload_test_data(SAMPLE_DATA)
        
        # Train model
        train_request = {
            "file_id": file_id,
            "target_column": "target",
            "model_type": "random_forest_classifier",
            "session_id": self.session_id,
            "test_size": 0.3,
            "use_cross_validation": True,
            "cv_folds": 3
        }
        
        response = client.post("/api/v1/training/train", json=train_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_type"] == "random_forest_classifier"
        assert data["is_classifier"] is True
        assert "train_metrics" in data
        assert "test_metrics" in data
        assert "cv_scores" in data
        
        print("✅ Classification model training test passed")
        return file_id
    
    def test_train_regression_model(self):
        """Test training a regression model"""
        # Upload test data
        file_id = self.upload_test_data(REGRESSION_DATA)
        
        # Train model
        train_request = {
            "file_id": file_id,
            "target_column": "target",
            "model_type": "random_forest_regressor",
            "session_id": "regression_session",
            "test_size": 0.3,
            "use_cross_validation": True,
            "cv_folds": 3
        }
        
        response = client.post("/api/v1/training/train", json=train_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_type"] == "random_forest_regressor"
        assert data["is_classifier"] is False
        assert "train_metrics" in data
        assert "test_metrics" in data
        
        print("✅ Regression model training test passed")
        return file_id
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning"""
        # Upload test data
        file_id = self.upload_test_data(SAMPLE_DATA)
        
        # Hyperparameter tuning request
        tuning_request = {
            "file_id": file_id,
            "target_column": "target",
            "model_type": "random_forest_classifier",
            "param_grid": {
                "n_estimators": [10, 20],
                "max_depth": [3, 5]
            },
            "cv_folds": 3,
            "session_id": "tuning_session"
        }
        
        response = client.post("/api/v1/training/hyperparameter-tuning", json=tuning_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "best_params" in data
        assert "best_score" in data
        assert "cv_results" in data
        
        print("✅ Hyperparameter tuning test passed")
    
    def test_model_info(self):
        """Test getting model information"""
        # First train a model
        self.test_train_classification_model()
        
        # Get model info
        response = client.get(f"/api/v1/training/model-info/{self.session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "model_type" in data
        assert "is_classifier" in data
        assert data["session_id"] == self.session_id
        
        print("✅ Model info test passed")
    
    def test_training_history(self):
        """Test getting training history"""
        # First train a model
        self.test_train_classification_model()
        
        # Get training history
        response = client.get(f"/api/v1/training/training-history/{self.session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "training_history" in data
        assert "total_trainings" in data
        assert data["total_trainings"] >= 1
        
        print("✅ Training history test passed")
    
    def test_make_predictions(self):
        """Test making predictions with a trained model"""
        # First train a model
        file_id = self.test_train_classification_model()
        
        # Create prediction data (same as training data but without target column)
        prediction_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "feature3": [0, 1, 0, 1, 0]
        }
        
        # Upload prediction data
        prediction_file_id = self.upload_test_data(prediction_data)
        
        # Make predictions
        prediction_request = {
            "file_id": prediction_file_id,
            "session_id": self.session_id
        }
        
        response = client.post("/api/v1/training/predict", json=prediction_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        
        print("✅ Predictions test passed")
    
    def test_save_model(self):
        """Test saving a trained model"""
        # First train a model
        self.test_train_classification_model()
        
        # Save model
        response = client.post(f"/api/v1/training/save-model/{self.session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "file_path" in data
        
        print("✅ Model saving test passed")
    
    def test_export_model(self):
        """Test exporting a trained model as base64"""
        # First train a model
        self.test_train_classification_model()
        
        # Export model
        response = client.get(f"/api/v1/training/export-model/{self.session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "model_data" in data
        assert "model_type" in data
        assert "size_bytes" in data
        
        print("✅ Model export test passed")
    
    def test_list_sessions(self):
        """Test listing all training sessions"""
        # First train a model to ensure we have at least one session
        self.test_train_classification_model()
        
        # List sessions
        response = client.get("/api/v1/training/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "sessions" in data
        assert "total_sessions" in data
        assert data["total_sessions"] >= 1
        
        print("✅ Session listing test passed")
    
    def test_delete_session(self):
        """Test deleting a training session"""
        # First train a model
        self.test_train_classification_model()
        
        # Delete session
        response = client.delete(f"/api/v1/training/session/{self.session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        
        # Verify session is deleted
        response = client.get(f"/api/v1/training/model-info/{self.session_id}")
        assert response.status_code == 404
        
        print("✅ Session deletion test passed")
    
    def test_multiple_model_types(self):
        """Test training multiple different model types"""
        models_to_test = [
            "logistic_regression",
            "decision_tree_classifier", 
            "svc",
            "linear_regression",
            "ridge"
        ]
        
        file_id = self.upload_test_data(SAMPLE_DATA)
        regression_file_id = self.upload_test_data(REGRESSION_DATA)
        
        for i, model_type in enumerate(models_to_test):
            session_id = f"multi_test_{i}"
            
            # Determine if it's classification or regression
            is_regression = model_type in ["linear_regression", "ridge"]
            target_data = REGRESSION_DATA if is_regression else SAMPLE_DATA
            current_file_id = regression_file_id if is_regression else file_id
            
            train_request = {
                "file_id": current_file_id,
                "target_column": "target",
                "model_type": model_type,
                "session_id": session_id,
                "test_size": 0.3
            }
            
            response = client.post("/api/v1/training/train", json=train_request)
            
            assert response.status_code == 200, f"Failed to train {model_type}: {response.text}"
            data = response.json()
            assert data["success"] is True
            assert data["model_type"] == model_type
            
        print(f"✅ Multiple model types test passed - Tested {len(models_to_test)} models")
    
    def test_error_handling(self):
        """Test error handling for various scenarios"""
        
        # Test with invalid session ID
        response = client.get("/api/v1/training/model-info/invalid_session")
        assert response.status_code == 404
        
        # Test with invalid file ID
        train_request = {
            "file_id": "invalid_file_id",
            "target_column": "target",
            "model_type": "random_forest_classifier",
            "session_id": "error_test_session"
        }
        response = client.post("/api/v1/training/train", json=train_request)
        assert response.status_code == 404
        
        # Test with invalid model type (this should be caught by FastAPI validation)
        invalid_request = {
            "file_id": "some_id",
            "target_column": "target",
            "model_type": "invalid_model_type",
            "session_id": "error_test_session"
        }
        response = client.post("/api/v1/training/train", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        print("✅ Error handling test passed")


def run_all_tests():
    """Run all tests in sequence"""
    print("🧪 Starting Training Endpoints Test Suite")
    print("=" * 50)
    
    test_instance = TestTrainingEndpoints()
    test_instance.setup_method()
    
    # Run tests in order
    try:
        test_instance.test_get_available_models()
        test_instance.test_get_available_models_simple()
        test_instance.test_get_model_hyperparameters()
        test_instance.test_train_classification_model()
        test_instance.test_train_regression_model()
        test_instance.test_hyperparameter_tuning()
        test_instance.test_model_info()
        test_instance.test_training_history()
        test_instance.test_make_predictions()
        test_instance.test_save_model()
        test_instance.test_export_model()
        test_instance.test_list_sessions()
        test_instance.test_multiple_model_types()
        test_instance.test_error_handling()
        test_instance.test_delete_session()  # Run this last as it deletes the session
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()
