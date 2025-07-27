"""
DVC Service for automated data and model versioning
Supports Azure Blob Storage backend
"""
import os
import subprocess
import asyncio
import hashlib
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.config import get_settings
import uuid
from sqlalchemy.orm import Session
from pymongo.database import Database

from app.models.sql_models import User, Project, ModelVersion, DatasetVersion
from app.models.mongo_schemas import DVCMetadataDocument


class DVCService:
    """
    Service for managing DVC operations with Azure Blob Storage
    Provides automated versioning without manual DVC commands
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_path = "dvc_storage"
        self.dvc_remote_url = self.settings.dvc_remote_url
        self.dvc_remote_name = self.settings.dvc_remote_name
        self.azure_connection_string = self.settings.dvc_azure_connection_string
        self.azure_container_name = self.settings.dvc_azure_container_name
        self.azure_storage_account = self.settings.azure_storage_account
        self.azure_storage_key = self.settings.azure_storage_key
        
        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)
    
    def _init_dvc_if_needed(self):
        """Initialize DVC repository if not already initialized"""
        try:
            if not os.path.exists('.dvc'):
                # Initialize DVC
                subprocess.run(['dvc', 'init'], check=True, capture_output=True, text=True)
                
            # Configure Azure remote if provided
            if self.dvc_remote_url and (self.azure_connection_string or (self.azure_storage_account and self.azure_storage_key)):
                # Remove existing remote if it exists
                subprocess.run(['dvc', 'remote', 'remove', self.dvc_remote_name], 
                             capture_output=True, text=True)
                
                # Add Azure remote
                subprocess.run([
                    'dvc', 'remote', 'add', '-d', self.dvc_remote_name, self.dvc_remote_url
                ], check=True, capture_output=True, text=True)
                
                # Configure Azure authentication
                if self.azure_connection_string:
                    # Use connection string (preferred for containers)
                    subprocess.run([
                        'dvc', 'remote', 'modify', self.dvc_remote_name, 
                        'connection_string', self.azure_connection_string
                    ], check=True, capture_output=True, text=True)
                elif self.azure_storage_account and self.azure_storage_key:
                    # Use account name and key (alternative for containers)
                    subprocess.run([
                        'dvc', 'remote', 'modify', self.dvc_remote_name, 
                        'account_name', self.azure_storage_account
                    ], check=True, capture_output=True, text=True)
                    
                    subprocess.run([
                        'dvc', 'remote', 'modify', self.dvc_remote_name, 
                        'account_key', self.azure_storage_key
                    ], check=True, capture_output=True, text=True)
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: DVC initialization failed: {e}")
        except Exception as e:
            print(f"Warning: DVC setup error: {e}")
    
    def _configure_remote(self):
        """Configure DVC remote storage"""
        try:
            # Add remote
            subprocess.run([
                "dvc", "remote", "add", "-d", self.dvc_remote_name, self.dvc_remote_url
            ], check=True, capture_output=True)
            
            print(f"✅ DVC remote '{self.dvc_remote_name}' configured: {self.dvc_remote_url}")
            
        except subprocess.CalledProcessError as e:
            # Remote might already exist
            if "already exists" in str(e.stderr):
                print(f"✅ DVC remote '{self.dvc_remote_name}' already configured")
            else:
                print(f"❌ Failed to configure DVC remote: {e}")
    
    def get_user_project_path(self, user_id: str, project_id: str, data_type: str = "models") -> Path:
        """Get isolated storage path for user and project"""
        path = Path(self.base_path) / data_type / user_id / project_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    async def version_model(
        self,
        model_path: str,
        user_id: str,
        project_id: str,
        model_name: str,
        version: str,
        db_session: Session,
        mongo_db: Database,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Version a model with DVC and store metadata"""
        
        try:
            # Create user/project specific path
            storage_path = self.get_user_project_path(user_id, project_id, "models")
            versioned_model_path = storage_path / f"{model_name}_v{version}"
            versioned_model_path.mkdir(exist_ok=True)
            
            # Copy model to versioned location
            model_file = versioned_model_path / "model.joblib"
            shutil.copy2(model_path, model_file)
            
            # Create metadata file
            model_metadata = {
                "name": model_name,
                "version": version,
                "user_id": user_id,
                "project_id": project_id,
                "created_at": datetime.utcnow().isoformat(),
                "original_path": model_path,
                "metadata": metadata or {}
            }
            
            metadata_file = versioned_model_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(model_metadata, f, indent=2)
            
            # Add to DVC
            dvc_file_path = await self._add_to_dvc(str(versioned_model_path))
            
            # Calculate file hash and size
            file_hash = self._calculate_directory_hash(str(versioned_model_path))
            file_size = self._get_directory_size(str(versioned_model_path))
            
            # Store in PostgreSQL
            model_version_record = ModelVersion(
                project_id=project_id,
                name=model_name,
                version=version,
                model_type=metadata.get("model_type", "unknown") if metadata else "unknown",
                storage_path=str(versioned_model_path),
                dvc_path=dvc_file_path,
                size_bytes=file_size,
                checksum=file_hash,
                performance_metrics=metadata.get("performance_metrics", {}) if metadata else {}
            )
            
            db_session.add(model_version_record)
            db_session.commit()
            db_session.refresh(model_version_record)
            
            # Store DVC metadata in MongoDB
            dvc_metadata = DVCMetadataDocument(
                file_path=str(versioned_model_path),
                project_id=project_id,
                user_id=user_id,
                dvc_file_path=dvc_file_path,
                md5_hash=file_hash,
                size=file_size,
                file_type="model",
                version=version,
                tags=[model_name, f"user_{user_id}", f"project_{project_id}"],
                custom_metadata=metadata or {}
            )
            
            await mongo_db.dvc_metadata.insert_one(dvc_metadata.dict())
            
            # Push to remote if configured
            if self.dvc_remote_url:
                await self._push_to_remote(dvc_file_path)
            
            return {
                "success": True,
                "model_version_id": str(model_version_record.id),
                "storage_path": str(versioned_model_path),
                "dvc_path": dvc_file_path,
                "version": version,
                "hash": file_hash,
                "size_bytes": file_size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to version model: {str(e)}"
            }
    
    async def version_dataset(
        self,
        dataset_path: str,
        user_id: str,
        project_id: str,
        dataset_name: str,
        version: str,
        db_session: Session,
        mongo_db: Database,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Version a dataset with DVC and store metadata"""
        
        try:
            # Create user/project specific path
            storage_path = self.get_user_project_path(user_id, project_id, "datasets")
            versioned_dataset_path = storage_path / f"{dataset_name}_v{version}"
            versioned_dataset_path.mkdir(exist_ok=True)
            
            # Copy dataset to versioned location
            if os.path.isfile(dataset_path):
                dataset_file = versioned_dataset_path / Path(dataset_path).name
                shutil.copy2(dataset_path, dataset_file)
            else:
                shutil.copytree(dataset_path, versioned_dataset_path / Path(dataset_path).name)
            
            # Create metadata file
            dataset_metadata = {
                "name": dataset_name,
                "version": version,
                "user_id": user_id,
                "project_id": project_id,
                "created_at": datetime.utcnow().isoformat(),
                "original_path": dataset_path,
                "metadata": metadata or {}
            }
            
            metadata_file = versioned_dataset_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(dataset_metadata, f, indent=2)
            
            # Add to DVC
            dvc_file_path = await self._add_to_dvc(str(versioned_dataset_path))
            
            # Calculate file hash and size
            file_hash = self._calculate_directory_hash(str(versioned_dataset_path))
            file_size = self._get_directory_size(str(versioned_dataset_path))
            
            # Store in PostgreSQL
            dataset_version_record = DatasetVersion(
                project_id=project_id,
                name=dataset_name,
                version=version,
                storage_path=str(versioned_dataset_path),
                dvc_path=dvc_file_path,
                size_bytes=file_size,
                checksum=file_hash,
                num_rows=metadata.get("num_rows") if metadata else None,
                num_columns=metadata.get("num_columns") if metadata else None,
                schema_info=metadata.get("schema_info", {}) if metadata else {},
                statistics=metadata.get("statistics", {}) if metadata else {}
            )
            
            db_session.add(dataset_version_record)
            db_session.commit()
            db_session.refresh(dataset_version_record)
            
            # Store DVC metadata in MongoDB
            dvc_metadata = DVCMetadataDocument(
                file_path=str(versioned_dataset_path),
                project_id=project_id,
                user_id=user_id,
                dvc_file_path=dvc_file_path,
                md5_hash=file_hash,
                size=file_size,
                file_type="dataset",
                version=version,
                tags=[dataset_name, f"user_{user_id}", f"project_{project_id}"],
                custom_metadata=metadata or {}
            )
            
            await mongo_db.dvc_metadata.insert_one(dvc_metadata.dict())
            
            # Push to remote if configured
            if self.dvc_remote_url:
                await self._push_to_remote(dvc_file_path)
            
            return {
                "success": True,
                "dataset_version_id": str(dataset_version_record.id),
                "storage_path": str(versioned_dataset_path),
                "dvc_path": dvc_file_path,
                "version": version,
                "hash": file_hash,
                "size_bytes": file_size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to version dataset: {str(e)}"
            }
    
    async def _add_to_dvc(self, file_path: str) -> str:
        """Add file/directory to DVC tracking"""
        try:
            # Add to DVC
            result = subprocess.run(
                ["dvc", "add", file_path],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Return the .dvc file path
            dvc_file_path = f"{file_path}.dvc"
            return dvc_file_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to add to DVC: {e.stderr}")
    
    async def _push_to_remote(self, dvc_file_path: str):
        """Push DVC file to remote storage"""
        try:
            subprocess.run(
                ["dvc", "push", dvc_file_path],
                check=True,
                capture_output=True
            )
            print(f"✅ Pushed to remote: {dvc_file_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to push to remote: {e.stderr}")
    
    async def retrieve_version(
        self,
        user_id: str,
        project_id: str,
        name: str,
        version: str,
        data_type: str = "models"
    ) -> Optional[str]:
        """Retrieve a specific version from DVC"""
        try:
            storage_path = self.get_user_project_path(user_id, project_id, data_type)
            versioned_path = storage_path / f"{name}_v{version}"
            
            if versioned_path.exists():
                return str(versioned_path)
            
            # Try to pull from remote
            dvc_file_path = f"{versioned_path}.dvc"
            if os.path.exists(dvc_file_path):
                subprocess.run(["dvc", "pull", dvc_file_path], check=True)
                return str(versioned_path) if versioned_path.exists() else None
            
            return None
            
        except Exception as e:
            print(f"Failed to retrieve version: {e}")
            return None
    
    async def list_versions(
        self,
        user_id: str,
        project_id: str,
        mongo_db: Database,
        data_type: str = "models"
    ) -> List[Dict[str, Any]]:
        """List all versions for a user and project"""
        try:
            cursor = mongo_db.dvc_metadata.find({
                "user_id": user_id,
                "project_id": project_id,
                "file_type": data_type
            }).sort("created_at", -1)
            
            versions = []
            async for doc in cursor:
                versions.append({
                    "file_path": doc["file_path"],
                    "version": doc["version"],
                    "size": doc["size"],
                    "created_at": doc["created_at"],
                    "tags": doc.get("tags", []),
                    "custom_metadata": doc.get("custom_metadata", {})
                })
            
            return versions
            
        except Exception as e:
            print(f"Failed to list versions: {e}")
            return []
    
    def _calculate_directory_hash(self, dir_path: str) -> str:
        """Calculate hash for directory contents"""
        hash_md5 = hashlib.md5()
        
        for root, dirs, files in os.walk(dir_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_directory_size(self, dir_path: str) -> int:
        """Get total size of directory"""
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size
    
    async def cleanup_old_versions(
        self,
        user_id: str,
        project_id: str,
        mongo_db: Database,
        keep_latest: int = 5
    ):
        """Clean up old versions, keeping only the latest N versions"""
        try:
            # Get all versions for this user/project
            versions = await self.list_versions(user_id, project_id, mongo_db)
            
            if len(versions) <= keep_latest:
                return
            
            # Remove older versions
            versions_to_remove = versions[keep_latest:]
            
            for version in versions_to_remove:
                file_path = version["file_path"]
                dvc_file_path = f"{file_path}.dvc"
                
                # Remove from filesystem
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        shutil.rmtree(file_path)
                
                if os.path.exists(dvc_file_path):
                    os.remove(dvc_file_path)
                
                # Remove from MongoDB
                await mongo_db.dvc_metadata.delete_one({
                    "file_path": file_path,
                    "user_id": user_id,
                    "project_id": project_id
                })
            
            print(f"✅ Cleaned up {len(versions_to_remove)} old versions")
            
        except Exception as e:
            print(f"Failed to cleanup versions: {e}")
