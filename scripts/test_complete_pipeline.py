#!/usr/bin/env python3
"""
Complete EasyML Pipeline Testing Script
Tests the entire ML pipeline from data upload to deployment
"""
import requests
import json
import pandas as pd
import numpy as np
import tempfile
import os
import time
import sys
from datetime import datetime
from sklearn.datasets import make_classification, make_regression
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
WAIT_TIME = 2  # seconds between requests

class CompletePipelineTester:
    def __init__(self):
        self.base_url = f"{BASE_URL}{API_PREFIX}"
        self.session = requests.Session()
        self.access_token = None
        self.user_info = None
        self.project_id = None
        self.file_id = None
        self.model_session_id = None
        self.model_version_id = None
        
        # Test results tracking
        self.test_results = {
            "server_health": False,
            "user_registration": False,
            "user_login": False,
            "project_creation": False,
            "data_upload": False,
            "model_training": False,
            "model_deployment": False,
            "prediction_test": False,
            "pipeline_success": False
        }
        
    def log_step(self, step_name, message):
        """Log test step with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {step_name}: {message}")
    
    def test_server_health(self):
        """Test if server is running"""
        self.log_step("HEALTH", "Testing server health...")
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                self.log_step("HEALTH", "✅ Server is running")
                self.test_results["server_health"] = True
                return True
            else:
                self.log_step("HEALTH", f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_step("HEALTH", f"❌ Cannot connect to server: {e}")
            return False
    
    def register_and_login_user(self):
        """Register and login test user"""
        self.log_step("AUTH", "Registering test user...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_data = {
            "username": f"testuser_{timestamp}",
            "email": f"test_{timestamp}@example.com",
            "password": "testpassword123",
            "full_name": f"Test User {timestamp}"
        }
        
        try:
            # Register user
            response = self.session.post(f"{self.base_url}/auth/register", json=user_data)
            if response.status_code == 201:
                self.log_step("AUTH", "✅ User registered successfully")
                self.test_results["user_registration"] = True
            elif response.status_code == 400 and "already registered" in response.text.lower():
                self.log_step("AUTH", "✅ User already exists, proceeding...")
                self.test_results["user_registration"] = True
            else:
                self.log_step("AUTH", f"❌ Registration failed: {response.text}")
                return False
            
            # Login user
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"]
            }
            
            response = self.session.post(f"{self.base_url}/auth/login", data=login_data)
            if response.status_code != 200:
                self.log_step("AUTH", f"❌ Login failed: {response.text}")
                return False
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
            
            self.log_step("AUTH", "✅ User logged in successfully")
            self.test_results["user_login"] = True
            
            # Get user info
            response = self.session.get(f"{self.base_url}/auth/me")
            if response.status_code == 200:
                self.user_info = response.json()
                self.log_step("AUTH", f"✅ User info retrieved: {self.user_info['username']}")
            
            return True
            
        except Exception as e:
            self.log_step("AUTH", f"❌ Authentication failed: {e}")
            return False
    
    def create_test_project(self):
        """Create a test project"""
        self.log_step("PROJECT", "Creating test project...")
        
        project_data = {
            "name": f"Test Project {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Automated test project for pipeline testing",
            "project_type": "classification"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/projects/", json=project_data)
            if response.status_code != 201:
                self.log_step("PROJECT", f"❌ Project creation failed: {response.text}")
                return False
            
            project = response.json()
            self.project_id = project["id"]
            self.log_step("PROJECT", f"✅ Project created: {self.project_id}")
            self.test_results["project_creation"] = True
            return True
            
        except Exception as e:
            self.log_step("PROJECT", f"❌ Project creation failed: {e}")
            return False
    
    def upload_test_data(self):
        """Generate and upload test dataset"""
        self.log_step("DATA", "Generating and uploading test dataset...")
        
        try:
            # Generate synthetic classification dataset
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=42
            )
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Save to temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_file_path = f.name
            
            # Upload file
            with open(temp_file_path, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                data = {'project_id': self.project_id}
                
                response = self.session.post(
                    f"{self.base_url}/projects/{self.project_id}/upload",
                    files=files,
                    data=data
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            if response.status_code != 200:
                self.log_step("DATA", f"❌ Data upload failed: {response.text}")
                return False
            
            upload_result = response.json()
            self.file_id = upload_result.get("file_id")
            self.log_step("DATA", f"✅ Data uploaded successfully: {self.file_id}")
            self.log_step("DATA", f"   Dataset shape: {df.shape}")
            self.test_results["data_upload"] = True
            return True
            
        except Exception as e:
            self.log_step("DATA", f"❌ Data upload failed: {e}")
            return False
    
    def train_model(self):
        """Train a model using the uploaded data"""
        self.log_step("TRAIN", "Starting model training...")
        
        training_config = {
            "project_id": self.project_id,
            "target_column": "target",
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": 42
            },
            "test_size": 0.2,
            "cross_validation": True,
            "cv_folds": 3
        }
        
        try:
            response = self.session.post(f"{self.base_url}/training/train", json=training_config)
            if response.status_code != 200:
                self.log_step("TRAIN", f"❌ Training failed: {response.text}")
                return False
            
            training_result = response.json()
            self.model_session_id = training_result.get("session_id")
            
            self.log_step("TRAIN", f"✅ Training started: {self.model_session_id}")
            
            # Display training metrics
            if "test_metrics" in training_result:
                metrics = training_result["test_metrics"]
                self.log_step("TRAIN", f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                self.log_step("TRAIN", f"   F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
                self.log_step("TRAIN", f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
                self.log_step("TRAIN", f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            
            self.test_results["model_training"] = True
            return True
            
        except Exception as e:
            self.log_step("TRAIN", f"❌ Training failed: {e}")
            return False
    
    def check_model_versions(self):
        """Check if model versions were created"""
        self.log_step("MODEL", "Checking model versions...")
        
        try:
            # Wait a moment for model to be saved
            time.sleep(3)
            
            response = self.session.get(f"{self.base_url}/projects/{self.project_id}/models")
            if response.status_code != 200:
                self.log_step("MODEL", f"❌ Failed to get models: {response.text}")
                return False
            
            models = response.json()
            if not models:
                self.log_step("MODEL", "❌ No models found")
                return False
            
            self.log_step("MODEL", f"✅ Found {len(models)} model(s)")
            for model in models:
                self.log_step("MODEL", f"   Model: {model.get('name')} v{model.get('version')}")
                if model.get('status') == 'active':
                    self.model_version_id = model.get('id')
            
            return True
            
        except Exception as e:
            self.log_step("MODEL", f"❌ Model check failed: {e}")
            return False
    
    def test_deployment(self):
        """Test model deployment"""
        self.log_step("DEPLOY", "Testing model deployment...")
        
        if not self.model_version_id:
            self.log_step("DEPLOY", "❌ No active model version found")
            return False
        
        deployment_config = {
            "project_id": self.project_id,
            "environment": "development",
            "deployment_config": {
                "model_selection": {
                    "primary_metric": "f1_score",
                    "metric_threshold": 0.5,
                    "secondary_metrics": {},
                    "prefer_latest": True
                },
                "scaling": {
                    "min_replicas": 1,
                    "max_replicas": 3
                }
            },
            "dry_run": True  # Don't actually deploy for testing
        }
        
        try:
            response = self.session.post(f"{self.base_url}/deployment/deploy", json=deployment_config)
            if response.status_code != 200:
                self.log_step("DEPLOY", f"❌ Deployment test failed: {response.text}")
                return False
            
            deployment_result = response.json()
            self.log_step("DEPLOY", f"✅ Deployment test successful")
            self.log_step("DEPLOY", f"   Status: {deployment_result.get('status')}")
            self.log_step("DEPLOY", f"   Selected Model: {deployment_result.get('selected_model', {}).get('model_name')}")
            
            self.test_results["model_deployment"] = True
            return True
            
        except Exception as e:
            self.log_step("DEPLOY", f"❌ Deployment test failed: {e}")
            return False
    
    def test_prediction(self):
        """Test prediction with trained model"""
        self.log_step("PREDICT", "Testing model prediction...")
        
        # Generate test prediction data
        X_test, _ = make_classification(
            n_samples=5,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=123
        )
        
        prediction_data = {
            "session_id": self.model_session_id,
            "data": X_test.tolist()
        }
        
        try:
            response = self.session.post(f"{self.base_url}/training/predict", json=prediction_data)
            if response.status_code != 200:
                self.log_step("PREDICT", f"❌ Prediction failed: {response.text}")
                return False
            
            predictions = response.json()
            self.log_step("PREDICT", f"✅ Prediction successful")
            self.log_step("PREDICT", f"   Predictions: {predictions.get('predictions', [])[:3]}...")  # Show first 3
            
            self.test_results["prediction_test"] = True
            return True
            
        except Exception as e:
            self.log_step("PREDICT", f"❌ Prediction failed: {e}")
            return False
    
    def check_dvc_integration(self):
        """Check DVC integration"""
        self.log_step("DVC", "Checking DVC integration...")
        
        try:
            # Check if models are in DVC storage
            dvc_storage_path = Path("dvc_storage/models")
            if dvc_storage_path.exists():
                user_dirs = list(dvc_storage_path.iterdir())
                self.log_step("DVC", f"✅ DVC storage found with {len(user_dirs)} user directories")
                
                # Look for our user's models
                if self.user_info:
                    user_id = self.user_info.get('id')
                    user_path = dvc_storage_path / str(user_id)
                    if user_path.exists():
                        project_dirs = list(user_path.iterdir())
                        self.log_step("DVC", f"✅ User has {len(project_dirs)} project(s) in DVC")
                        return True
                
                self.log_step("DVC", "⚠️ User models not found in DVC storage")
                return True  # DVC is working, just no models yet
            else:
                self.log_step("DVC", "❌ DVC storage directory not found")
                return False
                
        except Exception as e:
            self.log_step("DVC", f"❌ DVC check failed: {e}")
            return False
    
    def run_complete_test(self):
        """Run the complete pipeline test"""
        print("=" * 60)
        print("🚀 EasyML Complete Pipeline Test")
        print("=" * 60)
        
        test_steps = [
            ("Server Health", self.test_server_health),
            ("User Auth", self.register_and_login_user),
            ("Project Creation", self.create_test_project),
            ("Data Upload", self.upload_test_data),
            ("Model Training", self.train_model),
            ("Model Versions", self.check_model_versions),
            ("Deployment Test", self.test_deployment),
            ("Prediction Test", self.test_prediction),
            ("DVC Integration", self.check_dvc_integration)
        ]
        
        passed_tests = 0
        total_tests = len(test_steps)
        
        for step_name, test_func in test_steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {step_name} PASSED")
                else:
                    print(f"❌ {step_name} FAILED")
                
                time.sleep(WAIT_TIME)  # Brief pause between tests
                
            except Exception as e:
                print(f"❌ {step_name} FAILED with exception: {e}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 PIPELINE TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 PIPELINE TEST SUCCESSFUL!")
            self.test_results["pipeline_success"] = True
        else:
            print("⚠️ PIPELINE TEST NEEDS ATTENTION")
        
        # Detailed results
        print("\nDetailed Results:")
        for key, value in self.test_results.items():
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"  {key.replace('_', ' ').title()}: {status}")
        
        print("\n" + "=" * 60)
        return self.test_results

def main():
    """Main test runner"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("EasyML Complete Pipeline Tester")
        print("Usage: python test_complete_pipeline.py")
        print("\nThis script tests the entire EasyML pipeline:")
        print("1. Server health check")
        print("2. User registration and authentication")
        print("3. Project creation")
        print("4. Data upload")
        print("5. Model training")
        print("6. Model versioning")
        print("7. Deployment testing")
        print("8. Prediction testing")
        print("9. DVC integration")
        return
    
    tester = CompletePipelineTester()
    results = tester.run_complete_test()
    
    # Exit with appropriate code
    exit_code = 0 if results["pipeline_success"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
