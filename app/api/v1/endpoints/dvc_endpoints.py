"""
DVC (Data Version Control) API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os

from app.core.auth import get_current_active_user
from app.core.database import get_session
from app.models.sql_models import User, Project
from app.services.dvc_service import DVCService

router = APIRouter(prefix="/dvc", tags=["Data Version Control"])


class VersionResponse(BaseModel):
    success: bool
    version_id: Optional[str] = None
    storage_path: Optional[str] = None
    dvc_path: Optional[str] = None
    version: Optional[str] = None
    hash: Optional[str] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None


class VersionListResponse(BaseModel):
    success: bool
    versions: List[Dict[str, Any]]
    total_count: int


@router.post("/projects/{project_id}/models/version", response_model=VersionResponse)
async def version_model(
    project_id: str,
    model_file: UploadFile = File(...),
    model_name: str = "model",
    metadata: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Version a model file with DVC"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as temp_file:
        content = await model_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Parse metadata if provided
        import json
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                parsed_metadata = {"raw_metadata": metadata}
        
        # Generate version
        from datetime import datetime
        version = f"v_{int(datetime.utcnow().timestamp())}"
        
        # Version the model
        result = await dvc_service.version_model(
            model_path=temp_file_path,
            user_id=str(current_user.id),
            project_id=project_id,
            model_name=model_name,
            version=version,
            db_session=db,
            metadata=parsed_metadata
        )
        
        return VersionResponse(**result)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post("/projects/{project_id}/datasets/version", response_model=VersionResponse)
async def version_dataset(
    project_id: str,
    dataset_file: UploadFile = File(...),
    dataset_name: str = "dataset",
    metadata: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Version a dataset file with DVC"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{dataset_file.filename.split('.')[-1]}") as temp_file:
        content = await dataset_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Parse metadata if provided
        import json
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                parsed_metadata = {"raw_metadata": metadata}
        
        # Add file info to metadata
        parsed_metadata.update({
            "original_filename": dataset_file.filename,
            "content_type": dataset_file.content_type,
            "file_size": len(content)
        })
        
        # Generate version
        from datetime import datetime
        version = f"v_{int(datetime.utcnow().timestamp())}"
        
        # Version the dataset
        result = await dvc_service.version_dataset(
            dataset_path=temp_file_path,
            user_id=str(current_user.id),
            project_id=project_id,
            dataset_name=dataset_name,
            version=version,
            db_session=db,
            metadata=parsed_metadata
        )
        
        return VersionResponse(**result)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.get("/projects/{project_id}/models/versions", response_model=VersionListResponse)
async def get_model_versions(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Get all model versions for a project"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    versions = await dvc_service.list_versions(
        user_id=str(current_user.id),
        project_id=project_id,
        data_type="models"
    )
    
    return VersionListResponse(
        success=True,
        versions=versions,
        total_count=len(versions)
    )


@router.get("/projects/{project_id}/datasets/versions", response_model=VersionListResponse)
async def get_dataset_versions(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Get all dataset versions for a project"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    versions = await dvc_service.list_versions(
        user_id=str(current_user.id),
        project_id=project_id,
        data_type="datasets"
    )
    
    return VersionListResponse(
        success=True,
        versions=versions,
        total_count=len(versions)
    )


@router.get("/projects/{project_id}/models/{name}/versions/{version}")
async def get_model_version(
    project_id: str,
    name: str,
    version: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Retrieve a specific model version"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    version_path = await dvc_service.retrieve_version(
        user_id=str(current_user.id),
        project_id=project_id,
        name=name,
        version=version,
        data_type="models"
    )
    
    if not version_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found"
        )
    
    return {
        "success": True,
        "version_path": version_path,
        "name": name,
        "version": version
    }


@router.delete("/projects/{project_id}/cleanup")
async def cleanup_old_versions(
    project_id: str,
    keep_latest: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Clean up old versions, keeping only the latest N versions"""
    
    # Verify user has access to project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.status == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    dvc_service = DVCService()
    
    await dvc_service.cleanup_old_versions(
        user_id=str(current_user.id),
        project_id=project_id,
        keep_latest=keep_latest
    )
    
    return {
        "success": True,
        "message": f"Cleaned up old versions, kept latest {keep_latest} versions"
    }


@router.get("/status")
async def get_dvc_status():
    """Get DVC status and configuration"""
    dvc_service = DVCService()
    
    return {
        "success": True,
        "dvc_initialized": os.path.exists(".dvc"),
        "remote_configured": bool(dvc_service.dvc_remote_url),
        "remote_name": dvc_service.dvc_remote_name,
        "remote_url": dvc_service.dvc_remote_url,
        "base_storage_path": dvc_service.base_path
    }
