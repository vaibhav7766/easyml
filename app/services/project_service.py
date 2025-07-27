"""
Project management service integrating PostgreSQL, MongoDB, and MLflow
"""
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from pymongo.database import Database

from app.models.sql_models import Project, User, MLExperiment, ModelVersion
from app.models.mongo_schemas import ProjectConfigDocument, AuditLogDocument
from app.core.database import get_session, get_database
import mlflow


class ProjectService:
    """Service for managing projects with multi-database integration"""
    
    def __init__(self):
        self.models_base_path = os.path.join(os.getcwd(), "models")
        self.datasets_base_path = os.path.join(os.getcwd(), "datasets")
        os.makedirs(self.models_base_path, exist_ok=True)
        os.makedirs(self.datasets_base_path, exist_ok=True)
    
    async def create_project(
        self, 
        db: Session, 
        mongo_db: Database,
        user: User,
        name: str,
        description: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None
    ) -> Project:
        """Create a new project with proper folder structure"""
        
        # Create project in PostgreSQL
        project = Project(
            name=name,
            description=description,
            owner_id=user.id
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        
        # Create project folders
        project_models_path = os.path.join(self.models_base_path, str(project.id))
        project_datasets_path = os.path.join(self.datasets_base_path, str(project.id))
        os.makedirs(project_models_path, exist_ok=True)
        os.makedirs(project_datasets_path, exist_ok=True)
        
        # Create MLflow experiment
        if not mlflow_experiment_name:
            mlflow_experiment_name = f"project_{project.id}_{name}"
        
        try:
            experiment_id = mlflow.create_experiment(
                name=mlflow_experiment_name,
                artifact_location=project_models_path
            )
        except Exception:
            # Experiment might already exist
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
        
        # Store project configuration in MongoDB
        project_config = ProjectConfigDocument(
            project_id=str(project.id),
            user_id=str(user.id),
            name=name,
            description=description,
            mlflow_experiment_name=mlflow_experiment_name,
            model_storage_path=project_models_path,
            dataset_storage_path=project_datasets_path
        )
        
        await mongo_db.project_configs.insert_one(project_config.dict())
        
        # Log project creation
        await self._log_action(
            mongo_db,
            user_id=str(user.id),
            project_id=str(project.id),
            action_type="project_created",
            resource_type="project",
            resource_id=str(project.id),
            new_values={"name": name, "description": description}
        )
        
        return project
    
    async def get_user_projects(
        self, 
        db: Session, 
        user: User
    ) -> List[Project]:
        """Get all projects for a user"""
        return db.query(Project).filter(
            Project.owner_id == user.id,
            Project.is_active == True
        ).all()
    
    async def get_project_by_id(
        self,
        db: Session,
        mongo_db: Database,
        project_id: str,
        user: User
    ) -> Optional[Dict[str, Any]]:
        """Get project with detailed information"""
        # Get from PostgreSQL
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == user.id
        ).first()
        
        if not project:
            return None
        
        # Get configuration from MongoDB
        config = await mongo_db.project_configs.find_one({
            "project_id": project_id
        })
        
        # Get experiment count
        experiment_count = db.query(MLExperiment).filter(
            MLExperiment.project_id == project_id
        ).count()
        
        # Get model count
        model_count = db.query(ModelVersion).filter(
            ModelVersion.project_id == project_id
        ).count()
        
        return {
            "project": project,
            "config": config,
            "stats": {
                "experiments": experiment_count,
                "models": model_count
            }
        }
    
    async def update_project(
        self,
        db: Session,
        mongo_db: Database,
        project_id: str,
        user: User,
        updates: Dict[str, Any]
    ) -> Optional[Project]:
        """Update project information"""
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == user.id
        ).first()
        
        if not project:
            return None
        
        old_values = {
            "name": project.name,
            "description": project.description
        }
        
        # Update PostgreSQL
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)
        
        db.commit()
        db.refresh(project)
        
        # Update MongoDB configuration
        await mongo_db.project_configs.update_one(
            {"project_id": project_id},
            {"$set": {**updates, "updated_at": datetime.utcnow()}}
        )
        
        # Log update
        await self._log_action(
            mongo_db,
            user_id=str(user.id),
            project_id=project_id,
            action_type="project_updated",
            resource_type="project",
            resource_id=project_id,
            old_values=old_values,
            new_values=updates
        )
        
        return project
    
    async def delete_project(
        self,
        db: Session,
        mongo_db: Database,
        project_id: str,
        user: User
    ) -> bool:
        """Soft delete project"""
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == user.id
        ).first()
        
        if not project:
            return False
        
        # Soft delete in PostgreSQL
        project.is_active = False
        db.commit()
        
        # Update MongoDB
        await mongo_db.project_configs.update_one(
            {"project_id": project_id},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )
        
        # Log deletion
        await self._log_action(
            mongo_db,
            user_id=str(user.id),
            project_id=project_id,
            action_type="project_deleted",
            resource_type="project",
            resource_id=project_id
        )
        
        return True
    
    async def get_project_models_path(self, project_id: str) -> str:
        """Get models storage path for project"""
        return os.path.join(self.models_base_path, project_id)
    
    async def get_project_datasets_path(self, project_id: str) -> str:
        """Get datasets storage path for project"""
        return os.path.join(self.datasets_base_path, project_id)
    
    async def _log_action(
        self,
        mongo_db: Database,
        user_id: str,
        project_id: str,
        action_type: str,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any] = None,
        new_values: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ):
        """Log action to audit trail"""
        log_entry = AuditLogDocument(
            action_id=str(uuid.uuid4()),
            user_id=user_id,
            project_id=project_id,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values or {},
            new_values=new_values or {},
            session_id=session_id
        )
        
        await mongo_db.audit_logs.insert_one(log_entry.dict())
