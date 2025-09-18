"""
Simple MongoDB Connection Test with SSL Configuration
"""
import os
import ssl
from pymongo import MongoClient
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()

def test_mongodb_with_ssl_config():
    """Test MongoDB with different SSL configurations"""
    mongo_url = settings.mongo_url
    mongo_db_name = settings.mongo_db_name

    print(f"🔍 Testing MongoDB Connection...")
    print(f"📡 URL: {mongo_url[:50]}...")
    print(f"🗄️  Database: {mongo_db_name}")
    
    # Try different SSL configurations
    ssl_configs = [
        {"name": "Default SSL", "params": {}},
        {"name": "SSL with cert validation disabled", "params": {"tlsAllowInvalidCertificates": True}},
        {"name": "SSL with hostname verification disabled", "params": {"tlsAllowInvalidHostnames": True}},
        {"name": "Both disabled", "params": {"tlsAllowInvalidCertificates": True, "tlsAllowInvalidHostnames": True}},
    ]
    
    for config in ssl_configs:
        try:
            print(f"\n🧪 Trying: {config['name']}")
            
            client = MongoClient(mongo_url, **config['params'], serverSelectionTimeoutMS=5000)
            
            # Test connection
            client.admin.command('ping')
            print("✅ Connection successful!")
            
            # Test database access
            db = client[mongo_db_name]
            collections = db.list_collection_names()
            print(f"📋 Collections: {collections if collections else 'None'}")
            
            # Test write
            test_collection = db.connection_test
            result = test_collection.insert_one({"test": "success", "config": config['name']})
            print(f"✅ Write test: OK (ID: {result.inserted_id})")
            
            # Clean up
            test_collection.delete_one({"_id": result.inserted_id})
            client.close()
            
            print(f"🎉 MongoDB connection working with: {config['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed with {config['name']}: {str(e)[:100]}...")
            
    return False

if __name__ == "__main__":
    success = test_mongodb_with_ssl_config()
    if success:
        print("\n🎉 MongoDB connection established!")
    else:
        print("\n❌ All MongoDB connection attempts failed.")
        print("💡 Possible solutions:")
        print("1. Check if your IP is whitelisted in MongoDB Atlas")
        print("2. Verify the connection string is correct")
        print("3. Check if the cluster is running")
        print("4. Try connecting from MongoDB Compass first")
