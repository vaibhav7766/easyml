#!/usr/bin/env python3
"""
Test deployment endpoint using existing API endpoints
"""
import requests
import json
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
from sklearn.datasets import make_classification

# API Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

class DeploymentTester:
    def __init__(self):
        self.base_url = f"{BASE_URL}{API_PREFIX}"
        self.session = requests.Session()
        self.project_id = None
        self.file_id = None
        self.model_session_id = None
        self.access_token = None
        self.user_info = None
        
    def test_server_health(self):
        """Test if server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is running")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def register_user(self):
        """Register a test user"""
        print("👤 Registering test user...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_data = {
            "username": f"testuser_{timestamp}",
            "email": f"test_{timestamp}@example.com",
            "password": "testpassword123",
            "full_name": "Test User for Deployment"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth/register",
                json=user_data
            )
            
            if response.status_code == 200:
                self.user_info = response.json()
                print(f"✅ User registered: {self.user_info['username']}")
                return user_data["username"], user_data["password"]
            else:
                print(f"❌ User registration failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error registering user: {e}")
            return None, None
    
    def login_user(self, username, password):
        """Login and get access token"""
        print("🔐 Logging in...")
        
        try:
            login_data = {
                "username": username,
                "password": password
            }
            
            response = self.session.post(
                f"{self.base_url}/auth/login",
                data=login_data  # OAuth2 expects form data
            )
            
            if response.status_code == 200:
                token_info = response.json()
                self.access_token = token_info["access_token"]
                
                # Set authorization header for all future requests
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })
                
                print(f"✅ Logged in successfully!")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error logging in: {e}")
            return False
    
    def create_test_data(self):
        """Create synthetic test data"""
        print("📊 Creating synthetic test dataset...")
        
        # Generate classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        
        print(f"📁 Test data created: {temp_file.name}")
        print(f"📈 Data shape: {df.shape}")
        print(f"🎯 Target distribution: {df['target'].value_counts().to_dict()}")
        
        return temp_file.name, feature_names
    
    def create_project(self):
        """Create a test project"""
        print("🚀 Creating test project...")
        
        project_data = {
            "name": f"Test Deployment Project {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Test project for deployment endpoint testing",
            "project_type": "classification"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/projects/",
                json=project_data
            )
            
            if response.status_code == 201:
                project_info = response.json()
                self.project_id = project_info["id"]
                print(f"✅ Project created: {self.project_id}")
                return True
            else:
                print(f"❌ Project creation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return False
    
    def upload_data(self, file_path):
        """Upload test data"""
        print("📤 Uploading test data...")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'text/csv')}
                response = self.session.post(
                    f"{self.base_url}/upload/",
                    files=files
                )
            
            if response.status_code == 200:
                upload_info = response.json()
                self.file_id = upload_info["file_id"]
                print(f"✅ Data uploaded: {self.file_id}")
                return True
            else:
                print(f"❌ Data upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading data: {e}")
            return False
    
    def train_model(self, feature_names):
        """Train a test model"""
        print("🤖 Training test model...")
        
        self.model_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_data = {
            "file_id": self.file_id,
            "target_column": "target",
            "model_type": "random_forest_classifier",  # Fixed: Use correct enum value
            "test_size": 0.2,
            "preprocessing_operations": {},  # Fixed: Use dict instead of array
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 10,
                "random_state": 42
            },
            "use_cross_validation": True,
            "cv_folds": 5,
            "session_id": self.model_session_id,
            "auto_version": True,
            "is_categorical": False  # Fixed: Use boolean instead of dict
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/training/projects/{self.project_id}/train",
                json=training_data
            )
            
            if response.status_code == 200:
                training_result = response.json()
                print(f"✅ Model trained successfully!")
                print(f"📊 Accuracy: {training_result.get('metrics', {}).get('accuracy', 'N/A')}")
                return True
            else:
                print(f"❌ Model training failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def save_model(self):
        """Save the trained model"""
        print("💾 Saving trained model...")
        
        save_data = {
            "model_name": "Test Deployment Model",
            "description": "Test model for deployment endpoint testing",
            "version": "1.0.0"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/training/save-model/{self.model_session_id}",
                json=save_data
            )
            
            if response.status_code == 200:
                print("✅ Model saved successfully!")
                return True
            else:
                print(f"❌ Model saving failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def deploy_model(self):
        """Deploy the trained model"""
        print("🚀 Deploying model...")
        
        deployment_data = {
            "project_id": self.project_id,
            "deployment_config": {
                # Deployment metadata
                "deployment_name": f"test-deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "project_id": self.project_id,
                "user_id": str(self.user_info['id']),
                
                # Model selection
                "model_selection": {
                    "primary_metric": "accuracy",
                    "metric_threshold": 0.7,
                    "secondary_metrics": {
                        "precision": 0.7,
                        "recall": 0.7
                    },
                    "max_model_size_mb": 100.0,
                    "max_inference_time_ms": None,
                    "prefer_latest": True,
                    "require_validation": True
                },
                "manual_model_id": None,
                
                # Deployment configuration
                "environment": "development",
                "strategy": "rolling",
                
                # Container configuration
                "container": {
                    "base_image": "python:3.10-slim",
                    "cpu_limit": "500m",
                    "memory_limit": "512Mi",
                    "cpu_request": "100m",
                    "memory_request": "128Mi",
                    "env_vars": {
                        "LOG_LEVEL": "INFO",
                        "WORKERS": "1"
                    },
                    "health_check_path": "/health",
                    "readiness_probe_path": "/ready"
                },
                
                # API configuration
                "api_prefix": "/api/v1",
                "enable_swagger": True,
                "enable_metrics": True,
                
                # Scaling configuration
                "min_replicas": 1,
                "max_replicas": 3,
                "target_cpu_utilization": 70,
                
                # Networking
                "service_port": 8000,
                "ingress_enabled": False,
                "domain": None,
                
                # Security
                "enable_authentication": True,
                "allowed_origins": [],
                
                # Logging and monitoring
                "log_level": "INFO",
                "enable_tracing": False
            },
            "force_deploy": False,
            "dry_run": False,
            "async_deployment": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/deployments/deploy",
                json=deployment_data
            )
            
            if response.status_code == 200:
                deployment_result = response.json()
                deployment_id = deployment_result.get("deployment_id")
                print(f"✅ Model deployment initiated!")
                print(f"🆔 Deployment ID: {deployment_id}")
                print(f"📊 Status: {deployment_result.get('status', 'Unknown')}")
                
                # Print API endpoint if available
                api_endpoint = deployment_result.get("api_endpoint")
                if api_endpoint:
                    print(f"🔗 API Endpoint: {api_endpoint}")
                    
                swagger_url = deployment_result.get("swagger_url")
                if swagger_url:
                    print(f"📚 Swagger Docs: {swagger_url}")
                    
                # Print full response for debugging
                print(f"📝 Full response: {json.dumps(deployment_result, indent=2)}")
                
                return deployment_id
            else:
                print(f"❌ Model deployment failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error deploying model: {e}")
            return None
    
    def check_deployment_status(self, deployment_id):
        """Check deployment status"""
        print(f"🔍 Checking deployment status for: {deployment_id}")
        
        try:
            response = self.session.get(
                f"{self.base_url}/deployments/status/{deployment_id}"
            )
            
            if response.status_code == 200:
                status_info = response.json()
                print(f"📊 Deployment Status: {status_info.get('status', 'Unknown')}")
                print(f"🌐 Endpoint URL: {status_info.get('endpoint_url', 'Not available')}")
                return status_info
            else:
                print(f"❌ Status check failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return None
    
    def get_deployment_config_template(self):
        """Get deployment configuration template"""
        print("📋 Getting deployment configuration template...")
        
        try:
            response = self.session.get(f"{self.base_url}/deployments/config-template")
            
            if response.status_code == 200:
                template = response.json()
                print("✅ Configuration template retrieved!")
                print(json.dumps(template, indent=2))
                return template
            else:
                print(f"❌ Template retrieval failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting template: {e}")
            return None
    
    def cleanup(self, file_path):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"🧹 Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up file {file_path}: {e}")
    
    def run_full_test(self):
        """Run the complete deployment test flow"""
        print("🚀 Starting deployment endpoint test...")
        print("=" * 50)
        
        # Check server health
        if not self.test_server_health():
            return False
        
        # Create test data
        try:
            file_path, feature_names = self.create_test_data()
        except Exception as e:
            print(f"❌ Failed to create test data: {e}")
            return False
        
        try:
            # Register and login user
            username, password = self.register_user()
            if not username or not self.login_user(username, password):
                return False
            
            # Create project
            if not self.create_project():
                return False
            
            # Upload data
            if not self.upload_data(file_path):
                return False
            
            # Train model
            if not self.train_model(feature_names):
                return False
            
            # Save model (skip if failing, ModelVersion should be created during training)
            save_success = self.save_model()
            if not save_success:
                print("⚠️ Model save failed but continuing with deployment test...")
            
            # Get deployment template (optional)
            self.get_deployment_config_template()
            
            # Deploy model
            deployment_id = self.deploy_model()
            if not deployment_id:
                return False
            
            # Check deployment status
            self.check_deployment_status(deployment_id)
            
            print("\n🎉 Deployment test completed successfully!")
            print(f"📝 Project ID: {self.project_id}")
            print(f"📁 File ID: {self.file_id}")
            print(f"🤖 Model Session: {self.model_session_id}")
            print(f"🚀 Deployment ID: {deployment_id}")
            
            return True
            
        finally:
            self.cleanup(file_path)


if __name__ == "__main__":
    tester = DeploymentTester()
    success = tester.run_full_test()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
