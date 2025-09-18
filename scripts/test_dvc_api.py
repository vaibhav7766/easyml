#!/usr/bin/env python3
"""
Test script for DVC API endpoints
"""
import asyncio
import aiohttp
import json
import os
import tempfile
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

class DVCEndpointTester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = None
        self.auth_token = None
        self.project_id = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def register_test_user(self):
        """Register a test user"""
        user_data = {
            "username": "test_dvc_user",
            "email": "test@dvc.com", 
            "password": "testpass123",
            "full_name": "DVC Test User"
        }
        
        url = f"{self.base_url}{API_PREFIX}/auth/register"
        async with self.session.post(url, json=user_data) as response:
            if response.status == 200:
                result = await response.json()
                print("✅ Test user registered successfully")
                return result
            elif response.status == 400:
                print("ℹ️  Test user already exists")
                return {"success": True}
            else:
                print(f"❌ Failed to register user: {response.status}")
                return None
    
    async def login(self):
        """Login and get auth token"""
        login_data = {
            "username": "test_dvc_user",
            "password": "testpass123"
        }
        
        url = f"{self.base_url}{API_PREFIX}/auth/login"
        async with self.session.post(url, data=login_data) as response:
            if response.status == 200:
                result = await response.json()
                self.auth_token = result["access_token"]
                print("✅ Login successful")
                return True
            else:
                print(f"❌ Login failed: {response.status}")
                text = await response.text()
                print(f"Error: {text}")
                return False
    
    async def create_test_project(self):
        """Create a test project"""
        project_data = {
            "name": "DVC Test Project",
            "description": "Test project for DVC endpoints",
            "mlflow_experiment_name": "dvc_test_experiment"
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        url = f"{self.base_url}{API_PREFIX}/projects/"
        
        async with self.session.post(url, json=project_data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                self.project_id = result["id"]
                print(f"✅ Test project created: {self.project_id}")
                return True
            else:
                print(f"❌ Failed to create project: {response.status}")
                text = await response.text()
                print(f"Error: {text}")
                return False
    
    async def test_dvc_status(self):
        """Test DVC status endpoint"""
        print("\n🔍 Testing DVC Status Endpoint...")
        
        url = f"{self.base_url}{API_PREFIX}/dvc/status"
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print("✅ DVC Status endpoint working")
                print(f"   DVC Initialized: {result.get('dvc_initialized')}")
                print(f"   Remote Configured: {result.get('remote_configured')}")
                print(f"   Remote Name: {result.get('remote_name')}")
                print(f"   Base Storage Path: {result.get('base_storage_path')}")
                return True
            else:
                print(f"❌ DVC Status endpoint failed: {response.status}")
                return False
    
    async def test_version_model(self):
        """Test model versioning endpoint"""
        print("\n🔍 Testing Model Versioning Endpoint...")
        
        if not self.auth_token or not self.project_id:
            print("❌ Authentication or project required")
            return False
        
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
            temp_file.write(b"dummy_model_content_for_testing")
            temp_file_path = temp_file.name
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            url = f"{self.base_url}{API_PREFIX}/dvc/projects/{self.project_id}/models/version"
            
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('model_file', 
                          open(temp_file_path, 'rb'),
                          filename='test_model.joblib',
                          content_type='application/octet-stream')
            data.add_field('model_name', 'test_model')
            data.add_field('metadata', json.dumps({
                "algorithm": "test",
                "accuracy": 0.95,
                "test_run": True
            }))
            
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Model versioning endpoint working")
                    print(f"   Version ID: {result.get('version_id')}")
                    print(f"   Storage Path: {result.get('storage_path')}")
                    print(f"   Version: {result.get('version')}")
                    return True
                else:
                    print(f"❌ Model versioning failed: {response.status}")
                    text = await response.text()
                    print(f"Error: {text}")
                    return False
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def test_version_dataset(self):
        """Test dataset versioning endpoint"""
        print("\n🔍 Testing Dataset Versioning Endpoint...")
        
        if not self.auth_token or not self.project_id:
            print("❌ Authentication or project required")
            return False
        
        # Create a dummy dataset file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_file.write(b"col1,col2,col3\n1,2,3\n4,5,6\n")
            temp_file_path = temp_file.name
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            url = f"{self.base_url}{API_PREFIX}/dvc/projects/{self.project_id}/datasets/version"
            
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('dataset_file', 
                          open(temp_file_path, 'rb'),
                          filename='test_dataset.csv',
                          content_type='text/csv')
            data.add_field('dataset_name', 'test_dataset')
            data.add_field('metadata', json.dumps({
                "rows": 2,
                "columns": 3,
                "test_run": True
            }))
            
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Dataset versioning endpoint working")
                    print(f"   Version ID: {result.get('version_id')}")
                    print(f"   Storage Path: {result.get('storage_path')}")
                    print(f"   Version: {result.get('version')}")
                    return True
                else:
                    print(f"❌ Dataset versioning failed: {response.status}")
                    text = await response.text()
                    print(f"Error: {text}")
                    return False
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def test_list_versions(self):
        """Test listing versions endpoints"""
        print("\n🔍 Testing List Versions Endpoints...")
        
        if not self.auth_token or not self.project_id:
            print("❌ Authentication or project required")
            return False
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Test model versions
        url = f"{self.base_url}{API_PREFIX}/dvc/projects/{self.project_id}/models/versions"
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                print("✅ Model versions list endpoint working")
                print(f"   Total versions: {result.get('total_count', 0)}")
            else:
                print(f"❌ Model versions list failed: {response.status}")
        
        # Test dataset versions
        url = f"{self.base_url}{API_PREFIX}/dvc/projects/{self.project_id}/datasets/versions"
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                print("✅ Dataset versions list endpoint working")
                print(f"   Total versions: {result.get('total_count', 0)}")
                return True
            else:
                print(f"❌ Dataset versions list failed: {response.status}")
                return False
    
    async def run_all_tests(self):
        """Run all DVC endpoint tests"""
        print("🚀 Starting DVC Endpoint Tests...")
        print(f"Testing against: {self.base_url}")
        
        try:
            # Setup
            await self.register_test_user()
            if not await self.login():
                return False
            
            if not await self.create_test_project():
                return False
            
            # Test endpoints
            await self.test_dvc_status()
            await self.test_version_model()
            await self.test_version_dataset()
            await self.test_list_versions()
            
            print("\n🎉 All DVC endpoint tests completed!")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test function"""
    async with DVCEndpointTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    print("DVC API Endpoint Tester")
    print("=" * 50)
    
    # Check if server is running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8000))
    sock.close()
    
    if result != 0:
        print("❌ FastAPI server is not running on localhost:8000")
        print("Please start the server first:")
        print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        exit(1)
    
    print("✅ Server is running, starting tests...")
    asyncio.run(main())
