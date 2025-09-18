"""
Project management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pymongo.database import Database
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.core.auth import get_current_active_user
from app.core.database import get_db, get_database
from app.models.sql_models import User, Project
from app.services.project_service import ProjectService
from app.schemas.schemas import ProjectResponse, ProjectCreate as ProjectCreateSchema

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreateAPI(BaseModel):
    name: str
    description: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None


class ProjectDetailResponse(BaseModel):
    project: ProjectResponse
    config: Optional[Dict[str, Any]]
    stats: Dict[str, int]


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreateAPI,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    mongo_db: Optional[Database] = Depends(get_database)
):
    """Create a new project"""
    project_service = ProjectService()
    
    # Handle MongoDB unavailability gracefully
    if mongo_db is None:
        print("⚠️  Warning: MongoDB unavailable, creating project with PostgreSQL only")
    
    project = await project_service.create_project(
        db=db,
        mongo_db=mongo_db,
        user=current_user,
        name=project_data.name,
        description=project_data.description,
        mlflow_experiment_name=project_data.mlflow_experiment_name
    )
    
    # Convert to response model manually with proper UUID handling
    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        description=project.description,
        owner_id=str(project.owner_id),
        is_active=project.is_active,
        created_at=project.created_at,
        updated_at=project.updated_at
    )


@router.get("/", response_model=List[ProjectResponse])
async def get_user_projects(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all projects for the current user"""
    project_service = ProjectService()
    
    projects = await project_service.get_user_projects(
        db=db,
        user=current_user
    )
    
    return [ProjectResponse(**{
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
        "owner_id": str(project.owner_id),
        "is_active": project.is_active,
        "created_at": project.created_at,
        "updated_at": project.updated_at
    }) for project in projects]


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    mongo_db: Database = Depends(get_database)
):
    """Get project details"""
    project_service = ProjectService()
    
    project_data = await project_service.get_project_by_id(
        db=db,
        mongo_db=mongo_db,
        project_id=project_id,
        user=current_user
    )
    
    if not project_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return ProjectDetailResponse(
        project=ProjectResponse(**{
            "id": str(project_data["project"].id),
            "name": project_data["project"].name,
            "description": project_data["project"].description,
            "owner_id": str(project_data["project"].owner_id),
            "is_active": project_data["project"].is_active,
            "created_at": project_data["project"].created_at,
            "updated_at": project_data["project"].updated_at
        }),
        config=project_data["config"],
        stats=project_data["stats"]
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    mongo_db: Database = Depends(get_database)
):
    """Update project"""
    project_service = ProjectService()
    
    updates = project_update.dict(exclude_unset=True)
    
    project = await project_service.update_project(
        db=db,
        mongo_db=mongo_db,
        project_id=project_id,
        user=current_user,
        updates=updates
    )
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return ProjectResponse.from_orm(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    mongo_db: Database = Depends(get_database)
):
    """Delete project"""
    project_service = ProjectService()
    
    success = await project_service.delete_project(
        db=db,
        mongo_db=mongo_db,
        project_id=project_id,
        user=current_user
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/models-path")
async def get_project_models_path(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project models storage path"""
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    project_service = ProjectService()
    models_path = await project_service.get_project_models_path(project_id)
    datasets_path = await project_service.get_project_datasets_path(project_id)
    
    return {
        "models_path": models_path,
        "datasets_path": datasets_path
    }
