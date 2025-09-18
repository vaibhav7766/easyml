"""
Project management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from fastapi import HTTPException, status

from app.core.auth import get_current_active_user
from app.core.database import get_db
from app.models.sql_models import User, Project
from app.services.project_service import ProjectService
from app.schemas.schemas import ProjectResponse, ProjectCreate as ProjectCreateSchema

router = APIRouter(prefix="/projects", tags=["projects"])


def parse_uuid(id_str: str) -> UUID:
    try:
        return UUID(id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found"
        )

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

# class ProjectResponse(BaseModel):
#     id: str
#     name: str
#     description: Optional[str]
#     owner_id: str
#     status: str  # NEW instead of status
#     created_at: datetime
#     updated_at: datetime

#     class Config:
#         orm_mode = True



@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreateAPI,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new project"""
    project_service = ProjectService()
    
    project = await project_service.create_project(
        db=db,
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
        status=project.status,
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
        "status": project.status,
        "created_at": project.created_at,
        "updated_at": project.updated_at
    }) for project in projects]


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id:str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project details"""
    project_service = ProjectService()
    
    project_id = parse_uuid(project_id)
    
    project_data = await project_service.get_project_by_id(
        db=db,
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
            "status": project_data["project"].status,
            "created_at": project_data["project"].created_at,
            "updated_at": project_data["project"].updated_at
        }),
        config=project_data["config"],
        stats=project_data["stats"]
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id:str,
    project_update: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update project"""
    project_service = ProjectService()
    
    project_id = parse_uuid(project_id)
    
    updates = project_update.dict(exclude_unset=True)
    
    project = await project_service.update_project(
        db=db,
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
    project_id:str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete project"""
    project_service = ProjectService()
    
    project_id = parse_uuid(project_id)
    
    success = await project_service.delete_project(
        db=db,
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
    project_id:str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project models storage path"""
    project_id = parse_uuid(project_id)
    
    # Verify user has access to project\
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
