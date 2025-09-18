#!/usr/bin/env python3
"""
Start EasyML server and test DVC endpoints
"""
import uvicorn
import threading
import time
import requests
import json
from app.main import app

def start_server():
    """Start the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def test_dvc_endpoints():
    """Test DVC endpoints"""
    time.sleep(3)  # Wait for server to start
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing EasyML DVC API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Server health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: DVC Status (no auth required)
    try:
        response = requests.get(f"{base_url}/api/v1/dvc/status")
        if response.status_code == 200:
            result = response.json()
            print("✅ DVC Status endpoint working")
            print(f"   DVC Initialized: {result.get('dvc_initialized')}")
            print(f"   Remote Configured: {result.get('remote_configured')}")
            print(f"   Remote Name: {result.get('remote_name')}")
            print(f"   Base Storage Path: {result.get('base_storage_path')}")
        else:
            print(f"❌ DVC Status failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ DVC Status error: {e}")
    
    # Test 3: API Documentation
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API Documentation accessible at http://localhost:8000/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")
    
    # Test 4: List available endpoints
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ Root endpoint working")
            print(f"   API Features: {result.get('features', [])}")
            print(f"   API Prefix: {result.get('api_prefix')}")
            print(f"   Documentation: {result.get('docs')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    print("\n🎯 DVC Endpoints Available:")
    print("=" * 30)
    print("📋 Status: GET /api/v1/dvc/status")
    print("📦 Version Model: POST /api/v1/dvc/projects/{project_id}/models/version")
    print("📊 Version Dataset: POST /api/v1/dvc/projects/{project_id}/datasets/version") 
    print("📝 List Model Versions: GET /api/v1/dvc/projects/{project_id}/models/versions")
    print("📈 List Dataset Versions: GET /api/v1/dvc/projects/{project_id}/datasets/versions")
    print("🔍 Get Model Version: GET /api/v1/dvc/projects/{project_id}/models/{name}/versions/{version}")
    print("🧹 Cleanup Versions: DELETE /api/v1/dvc/projects/{project_id}/cleanup")
    
    print(f"\n🌐 Server running at: {base_url}")
    print(f"📚 API Documentation: {base_url}/docs")
    print(f"📖 ReDoc Documentation: {base_url}/redoc")
    
    print("\n🔐 Authentication Required:")
    print("Most DVC endpoints require authentication. Use:")
    print("1. Register: POST /api/v1/auth/register")
    print("2. Login: POST /api/v1/auth/login")
    print("3. Create Project: POST /api/v1/projects/")
    print("4. Use Bearer token in Authorization header")

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test endpoints
    test_dvc_endpoints()
    
    # Keep server running
    print("\n⏰ Server will keep running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
