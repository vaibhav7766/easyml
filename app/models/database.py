"""
Database models and CRUD operations
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from pymongo.database import Database

from app.core.enums import ProjectStatus, TaskType, ModelType


class ProjectModel:
    """Project model for database operations"""
    
    def __init__(self, database: Database):
        self.db = database
        self.collection = database["projects"]
    
    def create_project(self, project_data: Dict[str, Any]) -> str:
        """Create a new project"""
        project_doc = {
            "name": project_data["name"],
            "description": project_data.get("description"),
            "user_id": project_data["user_id"],
            "status": ProjectStatus.CREATED,
            "file_path": project_data.get("file_path"),
            "processed_file_path": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": {}
        }
        result = self.collection.insert_one(project_doc)
        return str(result.inserted_id)
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID"""
        try:
            project = self.collection.find_one({"_id": ObjectId(project_id)})
            if project:
                project["_id"] = str(project["_id"])
            return project
        except Exception:
            return None
    
    def get_projects_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all projects for a user"""
        projects = list(self.collection.find({"user_id": user_id}))
        for project in projects:
            project["_id"] = str(project["_id"])
        return projects
    
    def update_project(self, project_id: str, update_data: Dict[str, Any]) -> bool:
        """Update project"""
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = self.collection.update_one(
                {"_id": ObjectId(project_id)}, 
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    def delete_project(self, project_id: str) -> bool:
        """Delete project"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(project_id)})
            return result.deleted_count > 0
        except Exception:
            return False


class ModelResultModel:
    """Model results and metrics storage"""
    
    def __init__(self, database: Database):
        self.db = database
        self.collection = database["model_results"]
    
    def save_model_result(self, result_data: Dict[str, Any]) -> str:
        """Save model training results"""
        result_doc = {
            "project_id": result_data["project_id"],
            "user_id": result_data["user_id"],
            "model_type": result_data["model_type"],
            "task_type": result_data["task_type"],
            "target_column": result_data["target_column"],
            "feature_columns": result_data["feature_columns"],
            "metrics": result_data["metrics"],
            "model_path": result_data["model_path"],
            "hyperparameters": result_data.get("hyperparameters", {}),
            "created_at": datetime.utcnow()
        }
        result = self.collection.insert_one(result_doc)
        return str(result.inserted_id)
    
    def get_model_results(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all model results for a project"""
        results = list(self.collection.find({"project_id": project_id}))
        for result in results:
            result["_id"] = str(result["_id"])
        return results


class DatasetMetadataModel:
    """Dataset metadata storage"""
    
    def __init__(self, database: Database):
        self.db = database
        self.collection = database["dataset_metadata"]
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save dataset metadata"""
        metadata_doc = {
            "project_id": metadata["project_id"],
            "filename": metadata["filename"],
            "file_size": metadata["file_size"],
            "columns": metadata["columns"],
            "rows": metadata["rows"],
            "column_types": metadata.get("column_types", {}),
            "missing_values": metadata.get("missing_values", {}),
            "statistics": metadata.get("statistics", {}),
            "created_at": datetime.utcnow()
        }
        result = self.collection.insert_one(metadata_doc)
        return str(result.inserted_id)
    
    def get_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata by project ID"""
        metadata = self.collection.find_one({"project_id": project_id})
        if metadata:
            metadata["_id"] = str(metadata["_id"])
        return metadata


def create_project_dict(name: str, file_path: str, user_id: str) -> Dict[str, Any]:
    """Create project dictionary (legacy compatibility)"""
    return {
        "name": name,
        "file_path": file_path,
        "user_id": user_id
    }


def project_from_mongo(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB document to API-friendly dict (legacy compatibility)"""
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc
