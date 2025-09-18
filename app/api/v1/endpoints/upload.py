"""
File upload endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.services.file_service import FileService
from app.schemas.schemas import FileInfoResponse, DataPreviewResponse, ErrorResponse
from app.core.auth import get_current_active_user
from app.core.database import get_session
from app.models.sql_models import User, Project, DatasetVersion

router = APIRouter()
file_service = FileService()


def get_next_dataset_version(db: Session, project_id: str, dataset_name: str) -> str:
    """
    Get the next version number for a dataset
    V1 -> V2 -> V2.1 -> V2.2 etc.
    """
    # Get the latest version for this dataset
    latest_version = db.query(DatasetVersion).filter(
        DatasetVersion.project_id == project_id,
        DatasetVersion.name == dataset_name
    ).order_by(DatasetVersion.created_at.desc()).first()
    
    if not latest_version:
        return "V1"
    
    current_version = latest_version.version
    
    # If current version is V1, next is V2
    if current_version == "V1":
        return "V2"
    
    # If current version is V2, V3, etc., increment to V2.1, V3.1, etc.
    if "." not in current_version:
        return f"{current_version}.1"
    
    # If current version is V2.1, V2.2, etc., increment the minor version
    major, minor = current_version.rsplit(".", 1)
    return f"{major}.{int(minor) + 1}"


def create_dataset_version(db: Session, project_id: str, dataset_name: str, file_path: str, 
                         file_info: dict, tag: str = "raw data") -> DatasetVersion:
    """
    Create a new dataset version record
    """
    version = get_next_dataset_version(db, project_id, dataset_name)
    
    dataset_version = DatasetVersion(
        project_id=project_id,
        name=dataset_name,
        version=version,
        tag=tag,
        storage_path=file_path,
        size_bytes=file_info.get("size_bytes", 0),
        num_rows=file_info.get("num_rows", 0),
        num_columns=file_info.get("num_columns", 0),
        checksum=file_info.get("checksum", ""),
        schema_info=file_info.get("columns", {}),
        statistics=file_info.get("statistics", {})
    )
    
    db.add(dataset_version)
    db.commit()
    db.refresh(dataset_version)
    
    return dataset_version


@router.post("/", response_model=FileInfoResponse)
async def upload_file(
    file: UploadFile = File(...),
    project_id: str = Query(..., description="ID of the project to upload the file to"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Upload a data file (CSV, Excel, JSON, Parquet) to a specific project
    Creates a new dataset version (V1 for first upload)
    
    - **file**: The data file to upload
    - **project_id**: ID of the project to upload the file to
    """
    # Validate that the user has access to the project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or you don't have access to it"
        )
    
    result = await file_service.upload_file(file, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
    
    # Create dataset version record (V1 for initial upload)
    try:
        dataset_name = file.filename.rsplit('.', 1)[0]  # Remove extension
        dataset_version = create_dataset_version(
            db=db,
            project_id=project_id,
            dataset_name=dataset_name,
            file_path=result["file_path"],
            file_info=result["file_info"],
            tag="raw data"
        )
        
        # Add version info to response
        result["file_info"]["dataset_version"] = dataset_version.version
        result["file_info"]["dataset_tag"] = dataset_version.tag
        result["file_info"]["dataset_version_id"] = str(dataset_version.id)
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to create dataset version: {e}")
        # Don't fail the upload if versioning fails
    
    return FileInfoResponse(
        success=True,
        file_id=result["file_id"],
        original_filename=result["original_filename"],
        file_info=result["file_info"],
        message=f"File uploaded successfully as {result['file_info'].get('dataset_version', 'V1')} ({result['file_info'].get('dataset_tag', 'raw data')})",
        project_id=project_id
    )


@router.get("/preview/{file_id}", response_model=DataPreviewResponse)
async def get_file_preview(
    file_id: str,
    n_rows: int = Query(5, ge=1, le=100, description="Number of rows to preview"),
    project_id: Optional[str] = Query(None, description="Project ID to locate the file"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
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
    db: Session = Depends(get_session)
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
    db: Session = Depends(get_session)
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
    db: Session = Depends(get_session)
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
