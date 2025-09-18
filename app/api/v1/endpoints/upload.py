"""
File upload endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.services.file_service import FileService
from app.schemas.schemas import FileInfoResponse, DataPreviewResponse, ErrorResponse
from app.core.auth import get_current_active_user
from app.core.database import get_db
from app.models.sql_models import User, Project

router = APIRouter()
file_service = FileService()


@router.post("/", response_model=FileInfoResponse)
async def upload_file(
    file: UploadFile = File(...),
    project_id: str = Query(..., description="ID of the project to upload the file to"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a data file (CSV, Excel, JSON, Parquet) to a specific project
    
    - **file**: The data file to upload
    - **project_id**: ID of the project to upload the file to
    """
    # Validate that the user has access to the project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.is_active == True
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or you don't have access to it"
        )
    
    result = await file_service.upload_file(file, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
    
    return FileInfoResponse(
        success=True,
        file_id=result["file_id"],
        original_filename=result["original_filename"],
        file_info=result["file_info"],
        message="File uploaded successfully",
        project_id=project_id
    )


@router.get("/preview/{file_id}", response_model=DataPreviewResponse)
async def get_file_preview(
    file_id: str,
    n_rows: int = Query(5, ge=1, le=100, description="Number of rows to preview"),
    project_id: Optional[str] = Query(None, description="Project ID to locate the file"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a preview of uploaded data file
    
    - **file_id**: Unique identifier of the uploaded file
    - **n_rows**: Number of rows to preview (1-100)
    - **project_id**: Optional project ID to locate the file
    """
    # If project_id is provided, validate access
    if project_id:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id,
            Project.is_active == True
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have access to it"
            )
    
    result = await file_service.get_data_preview(file_id, n_rows, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    
    return DataPreviewResponse(
        success=True,
        preview=result["preview"],
        columns=result["columns"],
        dtypes=result["dtypes"],
        shape=result["shape"],
        preview_rows=result["preview_rows"]
    )


@router.get("/list")
async def list_files(
    project_id: Optional[str] = Query(None, description="Project ID to filter files by"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List uploaded files, optionally filtered by project
    
    - **project_id**: Optional project ID to filter files by
    """
    # If project_id is provided, validate access
    if project_id:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id,
            Project.is_active == True
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have access to it"
            )
    
    result = file_service.list_files(project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to list files"))
    
    return {
        "success": True,
        "files": result["files"],
        "total_files": result["total_files"],
        "project_id": project_id
    }


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    project_id: Optional[str] = Query(None, description="Project ID to locate the file"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete an uploaded file
    
    - **file_id**: Unique identifier of the file to delete
    - **project_id**: Optional project ID to locate the file
    """
    # If project_id is provided, validate access
    if project_id:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id,
            Project.is_active == True
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have access to it"
            )
    
    result = file_service.delete_file(file_id, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    
    return {
        "success": True,
        "message": result["message"]
    }


@router.get("/info/{file_id}")
async def get_file_info(
    file_id: str,
    project_id: Optional[str] = Query(None, description="Project ID to locate the file"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about an uploaded file
    
    - **file_id**: Unique identifier of the file
    - **project_id**: Optional project ID to locate the file
    """
    # If project_id is provided, validate access
    if project_id:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id,
            Project.is_active == True
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have access to it"
            )
    
    result = await file_service.load_data(file_id, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    
    return {
        "success": True,
        "file_id": file_id,
        "data_summary": result["data_summary"],
        "shape": result["shape"],
        "project_id": project_id
    }
