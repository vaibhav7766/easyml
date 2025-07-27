"""
System status checker for EasyML
Validates that all components are properly configured
"""
import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_file_structure():
    """Check if all required files and directories exist"""
    print("📁 Checking file structure...")
    
    required_files = [
        ".env",
        "app/main.py",
        "app/core/database.py", 
        "app/core/auth.py",
        "app/services/dvc_service.py",
        "app/services/enhanced_model_training.py",
        "app/api/v1/endpoints/auth.py",
        "app/api/v1/endpoints/projects.py", 
        "app/api/v1/endpoints/dvc_endpoints.py",
        "scripts/init_database.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "uploads",
        "models", 
        "datasets",
        "logs",
        "app/models",
        "app/services",
        "app/api/v1/endpoints"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
        return False
    
    print("   ✅ All required files and directories present")
    return True


def check_environment_config():
    """Check environment configuration"""
    print("⚙️  Checking environment configuration...")
    
    if not os.path.exists(".env"):
        print("   ❌ .env file not found")
        return False
    
    # Read .env file
    env_vars = {}
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value
    
    required_vars = [
        "POSTGRES_URL",
        "MONGO_URL", 
        "MONGO_DB_NAME",
        "SECRET_KEY",
        "DVC_REMOTE_URL",
        "DVC_AZURE_CONNECTION_STRING"
    ]
    
    optional_vars = [
        "AZURE_STORAGE_ACCOUNT",
        "AZURE_STORAGE_KEY"
    ]
    
    missing_vars = []
    placeholder_vars = []
    
    for var in required_vars:
        if var not in env_vars:
            missing_vars.append(var)
        elif var == "POSTGRES_URL" and "username:password" in env_vars[var]:
            placeholder_vars.append(var)
        elif var == "MONGO_URL" and "username:password" in env_vars[var]:
            placeholder_vars.append(var)
        elif var == "SECRET_KEY" and env_vars[var] == "your-super-secret-key-change-this-in-production":
            placeholder_vars.append(var)
        elif var == "DVC_REMOTE_URL" and "your-container-name" in env_vars[var]:
            placeholder_vars.append(var)
        elif var == "DVC_AZURE_CONNECTION_STRING" and ("your-azure-storage-connection-string" in env_vars[var] or "yourstorageaccount" in env_vars[var]):
            placeholder_vars.append(var)
    
    # Check optional container-specific variables
    container_auth_available = False
    if "AZURE_STORAGE_ACCOUNT" in env_vars and "AZURE_STORAGE_KEY" in env_vars:
        if not ("yourstorageaccount" in env_vars["AZURE_STORAGE_ACCOUNT"] or "XXXXX" in env_vars["AZURE_STORAGE_KEY"]):
            container_auth_available = True
    
    if missing_vars:
        print(f"   ❌ Missing environment variables: {missing_vars}")
        return False
    
    if placeholder_vars:
        print(f"   ⚠️  Environment variables with placeholder values: {placeholder_vars}")
        print("   Please update these with actual values before running the system")
        return False
    
    print("   ✅ Environment configuration looks good")
    if container_auth_available:
        print(f"   ✅ Container Azure authentication available")
    elif "DVC_AZURE_CONNECTION_STRING" not in placeholder_vars:
        print(f"   ℹ️  Using connection string authentication")
    
    return True


def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking Python dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "asyncpg",
        "motor",
        "pymongo",
        "pandas",
        ("sklearn", "scikit-learn"),  # Import name, package name
        "mlflow",
        "passlib",
        ("jose", "python-jose")  # Import name, package name
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if isinstance(package, tuple):
                import_name, package_name = package
                __import__(import_name)
            else:
                __import__(package)
        except ImportError:
            package_name = package[1] if isinstance(package, tuple) else package
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"   ❌ Missing packages: {missing_packages}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("   ✅ All required packages installed")
    return True


def check_dvc_installation():
    """Check if DVC is properly installed"""
    print("🔄 Checking DVC installation...")
    
    try:
        import subprocess
        result = subprocess.run(['dvc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ DVC installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ DVC not properly installed")
            return False
    except FileNotFoundError:
        print("   ❌ DVC command not found")
        print("   Run: pip install dvc[s3]")
        return False


async def check_database_connections():
    """Check database connections"""
    print("🗄️  Checking database connections...")
    
    try:
        from app.core.config import get_settings
        settings = get_settings()
        
        # Check if URLs are configured (not placeholders)
        if "username:password" in settings.postgres_url:
            print("   ⚠️  PostgreSQL URL contains placeholder values")
            return False
        
        if "username:password" in settings.mongo_url:
            print("   ⚠️  MongoDB URL contains placeholder values") 
            return False
        
        print("   ✅ Database URLs configured (connection testing requires actual databases)")
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking database configuration: {e}")
        return False


def generate_status_report():
    """Generate a comprehensive status report"""
    print("\n" + "="*60)
    print("🔍 EasyML System Status Report")
    print("="*60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Environment Config", check_environment_config), 
        ("Python Dependencies", check_python_dependencies),
        ("DVC Installation", check_dvc_installation),
        ("Database Config", lambda: asyncio.run(check_database_connections()))
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if check_func():
            passed += 1
    
    print("\n" + "="*60)
    print(f"📊 Status Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 System is ready! All checks passed.")
        print("\n📋 Next Steps:")
        print("1. Update database URLs in .env with your actual database connections")
        print("2. Run: python scripts/init_database.py")
        print("3. Start the application: uvicorn app.main:app --reload")
        print("4. Test the API: python tests/test_workflow.py")
    else:
        print("⚠️  System needs attention. Please fix the issues above.")
        
    print("\n📚 Documentation:")
    print("- API Documentation: docs/API_DOCUMENTATION.md")
    print("- Deployment Guide: docs/DEPLOYMENT_GUIDE.md") 
    print("- Interactive API Docs: http://localhost:8000/docs (when running)")
    
    return passed == total


if __name__ == "__main__":
    generate_status_report()
