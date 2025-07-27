"""
Test utilities for EasyML API testing
"""
import requests
import json
import os
from typing import Dict, Any, Optional
import time


class EasyMLTestClient:
    """Test client for EasyML API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.token = None
        self.user_id = None
        self.session = requests.Session()
    
    def register_user(self, username: str, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """Register a new user"""
        response = self.session.post(
            f"{self.base_url}/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name or f"{username} User"
            }
        )
        return response.json()
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and store token"""
        response = self.session.post(
            f"{self.base_url}/auth/login",
            data={
                "username": username,
                "password": password
            }
        )
        result = response.json()
        
        if response.status_code == 200 and "access_token" in result:
            self.token = result["access_token"]
            self.user_id = result.get("user", {}).get("id")
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
        
        return result
    
    def create_project(self, name: str, description: str = "", project_type: str = "classification") -> Dict[str, Any]:
        """Create a new project"""
        response = self.session.post(
            f"{self.base_url}/projects",
            json={
                "name": name,
                "description": description,
                "project_type": project_type
            }
        )
        return response.json()
    
    def list_projects(self) -> Dict[str, Any]:
        """List all projects"""
        response = self.session.get(f"{self.base_url}/projects")
        return response.json()
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload a file"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/upload/file",
                files=files
            )
        return response.json()
    
    def train_model(self, project_id: str, file_id: str, target_column: str, 
                   model_type: str = "random_forest", auto_version: bool = True) -> Dict[str, Any]:
        """Train a model with automatic versioning"""
        response = self.session.post(
            f"{self.base_url}/training/projects/{project_id}/train",
            json={
                "file_id": file_id,
                "target_column": target_column,
                "model_type": model_type,
                "test_size": 0.2,
                "use_cross_validation": True,
                "cv_folds": 5,
                "session_id": f"test_session_{int(time.time())}",
                "auto_version": auto_version
            }
        )
        return response.json()
    
    def version_model(self, project_id: str, model_file_path: str, 
                     model_name: str = "test_model", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Version a model file"""
        with open(model_file_path, 'rb') as f:
            files = {'model_file': f}
            data = {
                'model_name': model_name,
                'metadata': json.dumps(metadata or {})
            }
            response = self.session.post(
                f"{self.base_url}/dvc/projects/{project_id}/models/version",
                files=files,
                data=data
            )
        return response.json()
    
    def version_dataset(self, project_id: str, dataset_file_path: str, 
                       dataset_name: str = "test_dataset", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Version a dataset file"""
        with open(dataset_file_path, 'rb') as f:
            files = {'dataset_file': f}
            data = {
                'dataset_name': dataset_name,
                'metadata': json.dumps(metadata or {})
            }
            response = self.session.post(
                f"{self.base_url}/dvc/projects/{project_id}/datasets/version",
                files=files,
                data=data
            )
        return response.json()
    
    def list_model_versions(self, project_id: str) -> Dict[str, Any]:
        """List all model versions for a project"""
        response = self.session.get(
            f"{self.base_url}/dvc/projects/{project_id}/models/versions"
        )
        return response.json()
    
    def list_dataset_versions(self, project_id: str) -> Dict[str, Any]:
        """List all dataset versions for a project"""
        response = self.session.get(
            f"{self.base_url}/dvc/projects/{project_id}/datasets/versions"
        )
        return response.json()
    
    def get_dvc_status(self) -> Dict[str, Any]:
        """Get DVC status"""
        response = self.session.get(f"{self.base_url}/dvc/status")
        return response.json()
    
    def cleanup_versions(self, project_id: str, keep_latest: int = 3) -> Dict[str, Any]:
        """Clean up old versions"""
        response = self.session.delete(
            f"{self.base_url}/dvc/projects/{project_id}/cleanup?keep_latest={keep_latest}"
        )
        return response.json()


def create_sample_data(filename: str = "test_data.csv"):
    """Create a sample CSV file for testing"""
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
        'feature4': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename


def run_complete_test_workflow():
    """Run a complete test workflow"""
    print("🧪 Starting EasyML Complete Test Workflow")
    print("=" * 50)
    
    # Initialize client
    client = EasyMLTestClient()
    
    # Test data
    test_username = f"testuser_{int(time.time())}"
    test_email = f"test_{int(time.time())}@example.com"
    test_password = "testpass123"
    
    try:
        # 1. Register user
        print("1. Registering user...")
        register_result = client.register_user(test_username, test_email, test_password)
        print(f"   ✅ User registered: {register_result}")
        
        # 2. Login
        print("2. Logging in...")
        login_result = client.login(test_username, test_password)
        print(f"   ✅ Login successful: {login_result.get('user', {}).get('username')}")
        
        # 3. Create project
        print("3. Creating project...")
        project_result = client.create_project("Test ML Project", "Test project for automated workflow")
        project_id = project_result["project"]["id"]
        print(f"   ✅ Project created: {project_id}")
        
        # 4. Create sample data
        print("4. Creating sample data...")
        data_file = create_sample_data("workflow_test_data.csv")
        print(f"   ✅ Sample data created: {data_file}")
        
        # 5. Upload data
        print("5. Uploading data...")
        upload_result = client.upload_file(data_file)
        file_id = upload_result["file_id"]
        print(f"   ✅ Data uploaded: {file_id}")
        
        # 6. Train model with auto-versioning
        print("6. Training model with auto-versioning...")
        train_result = client.train_model(project_id, file_id, "target", "random_forest", True)
        print(f"   ✅ Model trained with DVC versioning: {train_result.get('dvc_info', {})}")
        
        # 7. List model versions
        print("7. Listing model versions...")
        versions_result = client.list_model_versions(project_id)
        print(f"   ✅ Model versions: {len(versions_result.get('versions', []))} found")
        
        # 8. Version dataset manually
        print("8. Versioning dataset manually...")
        dataset_version_result = client.version_dataset(
            project_id, 
            data_file, 
            "workflow_dataset",
            {"description": "Test dataset for workflow", "rows": 1000}
        )
        print(f"   ✅ Dataset versioned: {dataset_version_result.get('version')}")
        
        # 9. List dataset versions
        print("9. Listing dataset versions...")
        dataset_versions_result = client.list_dataset_versions(project_id)
        print(f"   ✅ Dataset versions: {len(dataset_versions_result.get('versions', []))} found")
        
        # 10. Get DVC status
        print("10. Checking DVC status...")
        dvc_status = client.get_dvc_status()
        print(f"   ✅ DVC Status: {dvc_status}")
        
        print("=" * 50)
        print("🎉 Complete test workflow successful!")
        print(f"   - User: {test_username}")
        print(f"   - Project: {project_id}")
        print(f"   - Model versions: {len(versions_result.get('versions', []))}")
        print(f"   - Dataset versions: {len(dataset_versions_result.get('versions', []))}")
        print("   - All operations completed with automated DVC versioning!")
        
    except Exception as e:
        print(f"❌ Test workflow failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists("workflow_test_data.csv"):
            os.remove("workflow_test_data.csv")


if __name__ == "__main__":
    # Run the complete test workflow
    run_complete_test_workflow()
