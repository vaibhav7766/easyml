
from typing import Optional, Dict, Any
from bson import ObjectId

def create_project_dict(name: str, file_path: str, user_id: str) -> Dict[str, Any]:
    return {
        "name": name,
        "file_path": file_path,
        "user_id": user_id
    }

def project_from_mongo(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Convert MongoDB document to API-friendly dict
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc
