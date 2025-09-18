"""
Test script for dataset versioning functionality
"""
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
PROJECT_ID = "your_project_id_here"  # Replace with actual project ID
USERNAME = "your_username"  # Replace with actual username
PASSWORD = "your_password"  # Replace with actual password

def get_auth_token(username: str, password: str) -> str:
    """Get authentication token"""
    response = requests.post(
        f"{BASE_URL}/auth/token",
        data={"username": username, "password": password}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    raise Exception(f"Failed to authenticate: {response.text}")

def test_dataset_versioning():
    """Test the dataset versioning functionality"""
    
    # Get auth token
    try:
        token = get_auth_token(USERNAME, PASSWORD)
        headers = {"Authorization": f"Bearer {token}"}
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # Test 1: Upload initial dataset (should create V1)
    print("🔄 Testing initial dataset upload (V1)...")
    
    # Create a sample CSV file for testing
    sample_data = """name,age,salary
John,25,50000
Jane,30,60000
Bob,35,70000"""
    
    test_file_path = Path("test_dataset.csv")
    test_file_path.write_text(sample_data)
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_dataset.csv", f, "text/csv")}
            response = requests.post(
                f"{BASE_URL}/upload/?project_id={PROJECT_ID}",
                files=files,
                headers=headers
            )
        
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ Upload successful: {upload_result['message']}")
            print(f"   Version: {upload_result['file_info'].get('dataset_version', 'Unknown')}")
            print(f"   Tag: {upload_result['file_info'].get('dataset_tag', 'Unknown')}")
            file_id = upload_result['file_id']
        else:
            print(f"❌ Upload failed: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()
    
    # Test 2: Apply preprocessing (should create V2)
    print("\n🔄 Testing preprocessing (V2)...")
    
    preprocessing_request = {
        "file_id": file_id,
        "operations": {
            "standardize": ["age", "salary"],
            "remove_duplicates": True
        },
        "is_categorical": {}
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/preprocessing/apply?project_id={PROJECT_ID}",
            json=preprocessing_request,
            headers=headers
        )
        
        if response.status_code == 200:
            preprocess_result = response.json()
            print(f"✅ Preprocessing successful")
            print(f"   Message: {preprocess_result.get('message', 'No message')}")
        else:
            print(f"❌ Preprocessing failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
    
    # Test 3: Apply more preprocessing (should create V2.1)
    print("\n🔄 Testing additional preprocessing (V2.1)...")
    
    preprocessing_request2 = {
        "file_id": file_id,
        "operations": {
            "normalize": ["age"],
            "remove_outliers": ["salary"]
        },
        "is_categorical": {}
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/preprocessing/apply?project_id={PROJECT_ID}",
            json=preprocessing_request2,
            headers=headers
        )
        
        if response.status_code == 200:
            preprocess_result = response.json()
            print(f"✅ Additional preprocessing successful")
            print(f"   Message: {preprocess_result.get('message', 'No message')}")
        else:
            print(f"❌ Additional preprocessing failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Additional preprocessing error: {e}")
    
    # Test 4: List dataset versions
    print("\n🔄 Testing dataset version listing...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/preprocessing/dataset-versions/{PROJECT_ID}",
            headers=headers
        )
        
        if response.status_code == 200:
            versions_result = response.json()
            print(f"✅ Retrieved {versions_result['total_versions']} dataset versions:")
            
            for version in versions_result['versions']:
                print(f"   - {version['name']} {version['version']} ({version['tag']})")
                print(f"     Rows: {version['num_rows']}, Columns: {version['num_columns']}")
                print(f"     Created: {version['created_at']}")
                print()
        else:
            print(f"❌ Failed to list versions: {response.text}")
            
    except Exception as e:
        print(f"❌ Version listing error: {e}")

if __name__ == "__main__":
    print("🧪 Dataset Versioning Test Script")
    print("=" * 50)
    print("⚠️  Make sure to update PROJECT_ID, USERNAME, and PASSWORD before running!")
    print("⚠️  Make sure the server is running on http://localhost:8000")
    print()
    
    # Uncomment the next line and update the variables to run the test
    # test_dataset_versioning()
    print("📝 Update the configuration variables and uncomment the test call to run.")