import os
from pymongo import MongoClient

def get_mongo_client():
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set in environment variables")
    return MongoClient(mongo_uri)

def SessionDep():
    client = get_mongo_client()
    db = client.get_default_database()  # or specify db name if needed
    try:
        yield db
    finally:
        client.close()
