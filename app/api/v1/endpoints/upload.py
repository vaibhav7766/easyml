"""
File upload endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.services.file_service import FileService
from app.services.dvc_service import DVCService
from app.schemas.schemas import FileInfoResponse, DataPreviewResponse, ErrorResponse
from app.core.auth import get_current_active_user
from app.core.database import get_session
from app.models.sql_models import User, Project, DatasetVersion

router = APIRouter()
file_service = FileService()
dvc_service = DVCService()


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


async def create_dataset_version_with_dvc(
    db: Session, 
    project_id: str, 
    dataset_name: str, 
    file_path: str, 
    file_info: dict, 
    user_id: str,
    tag: str = "raw data",
    set_as_current: bool = True
) -> dict:
    """
    Create a new dataset version record with DVC integration
    """
    version = get_next_dataset_version(db, project_id, dataset_name)
    
    # Create metadata for DVC versioning
    metadata = {
        "num_rows": file_info.get("num_rows", 0),
        "num_columns": file_info.get("num_columns", 0),
        "schema_info": file_info.get("columns", {}),
        "statistics": file_info.get("statistics", {}),
        "tag": tag,
        "original_filename": file_info.get("original_filename", "")
    }
    
    # Version the dataset with DVC
    dvc_result = await dvc_service.version_dataset(
        dataset_path=file_path,
        user_id=user_id,
        project_id=project_id,
        dataset_name=dataset_name,
        version=version,
        db_session=db,
        metadata=metadata
    )
    
    if not dvc_result.get("success", False):
        return {
            "success": False,
            "error": f"Failed to version dataset with DVC: {dvc_result.get('error', 'Unknown error')}"
        }
    
    # If this should be the current version, update previous versions
    if set_as_current:
        # Mark all previous versions as not current
        db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id,
            DatasetVersion.name == dataset_name,
            DatasetVersion.is_current == True
        ).update({DatasetVersion.is_current: False})
        
        # Mark this version as current
        dataset_version_record = db.query(DatasetVersion).filter(
            DatasetVersion.id == dvc_result["dataset_version_id"]
        ).first()
        
        if dataset_version_record:
            dataset_version_record.is_current = True
            db.commit()
            db.refresh(dataset_version_record)
    
    return {
        "success": True,
        "dataset_version": dataset_version_record if set_as_current else None,
        "dvc_result": dvc_result
    }


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
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or you don't have access to it"
        )
    
    result = await file_service.upload_file(file, project_id=project_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
    
    # Create dataset version record with DVC integration
    try:
        dataset_name = file.filename.rsplit('.', 1)[0]  # Remove extension
        
        # First, load the data to get comprehensive file info
        data_result = await file_service.load_data(result["file_path"])
        if data_result.get("success"):
            # Update file_info with data summary
            result["file_info"].update({
                "num_rows": data_result["shape"][0],
                "num_columns": data_result["shape"][1],
                "columns": data_result["data_summary"]["columns"],
                "statistics": data_result["data_summary"]
            })
        
        # Create versioned dataset with DVC
        version_result = await create_dataset_version_with_dvc(
            db=db,
            project_id=project_id,
            dataset_name=dataset_name,
            file_path=result["file_path"],
            file_info=result["file_info"],
            user_id=str(current_user.id),
            tag="raw data",
            set_as_current=True
        )
        
        if version_result.get("success"):
            dataset_version = version_result["dataset_version"]
            dvc_result = version_result["dvc_result"]
            
            # Add version info to response
            result["file_info"]["dataset_version"] = dataset_version.version
            result["file_info"]["dataset_tag"] = dataset_version.tag
            result["file_info"]["dataset_version_id"] = str(dataset_version.id)
            result["file_info"]["is_current"] = dataset_version.is_current
            result["file_info"]["dvc_path"] = dvc_result.get("dvc_path", "")
            result["file_info"]["versioned_path"] = dvc_result.get("storage_path", "")
        else:
            print(f"⚠️ Warning: Failed to create DVC dataset version: {version_result.get('error')}")
            # Fallback to basic versioning without DVC
            result["file_info"]["dataset_version"] = "V1"
            result["file_info"]["dataset_tag"] = "raw data"
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to create dataset version: {e}")
        # Don't fail the upload if versioning fails
        result["file_info"]["dataset_version"] = "V1"
        result["file_info"]["dataset_tag"] = "raw data"
    
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
            Project.owner_id == current_user.id
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
            Project.owner_id == current_user.id
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
            Project.owner_id == current_user.id
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
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have access to it"
            )
    
    current_dataset = db.query(DatasetVersion).filter(
        DatasetVersion.project_id == project_id,
        DatasetVersion.is_current == True
    ).first()
    
    if not current_dataset:
        # Fallback to latest version if no current version is set
        current_dataset = db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id
        ).order_by(DatasetVersion.created_at.desc()).first()
    
    if not current_dataset:
        raise HTTPException(
            status_code=404,
            detail="No dataset found for this project. Please upload a dataset first."
        )
    
    file_id = current_dataset.storage_path
    
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


@router.get("/datasets/{project_id}/versions")
async def list_dataset_versions(
    project_id: str,
    dataset_name: Optional[str] = Query(None, description="Filter by dataset name"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    List all dataset versions for a project
    
    - **project_id**: ID of the project
    - **dataset_name**: Optional filter by dataset name
    """
    # Validate project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found or you don't have access to it"
        )
    
    # Build query
    query = db.query(DatasetVersion).filter(DatasetVersion.project_id == project_id)
    
    if dataset_name:
        query = query.filter(DatasetVersion.name == dataset_name)
    
    versions = query.order_by(DatasetVersion.created_at.desc()).all()
    
    version_list = []
    for version in versions:
        version_info = {
            "id": str(version.id),
            "name": version.name,
            "version": version.version,
            "tag": version.tag,
            "is_current": version.is_current,
            "storage_path": version.storage_path,
            "dvc_path": version.dvc_path,
            "size_bytes": version.size_bytes,
            "num_rows": version.num_rows,
            "num_columns": version.num_columns,
            "created_at": version.created_at,
            "schema_info": version.schema_info,
            "statistics": version.statistics
        }
        version_list.append(version_info)
    
    return {
        "success": True,
        "project_id": project_id,
        "dataset_name": dataset_name,
        "versions": version_list,
        "total_versions": len(version_list)
    }


@router.post("/datasets/{project_id}/versions/{version_id}/set-current")
async def set_current_dataset_version(
    project_id: str,
    version_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Set a specific dataset version as the current active version
    
    - **project_id**: ID of the project
    - **version_id**: ID of the dataset version to set as current
    """
    # Validate project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found or you don't have access to it"
        )
    
    # Get the specific version
    target_version = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.project_id == project_id
    ).first()
    
    if not target_version:
        raise HTTPException(
            status_code=404,
            detail="Dataset version not found"
        )
    
    # Mark all versions of this dataset as not current
    db.query(DatasetVersion).filter(
        DatasetVersion.project_id == project_id,
        DatasetVersion.name == target_version.name,
        DatasetVersion.is_current == True
    ).update({DatasetVersion.is_current: False})
    
    # Mark the target version as current
    target_version.is_current = True
    db.commit()
    db.refresh(target_version)
    
    return {
        "success": True,
        "message": f"Dataset version {target_version.version} is now the current version",
        "current_version": {
            "id": str(target_version.id),
            "name": target_version.name,
            "version": target_version.version,
            "tag": target_version.tag,
            "is_current": target_version.is_current,
            "storage_path": target_version.storage_path
        }
    }


@router.get("/datasets/{project_id}/current")
async def get_current_dataset_version(
    project_id: str,
    dataset_name: str = Query(..., description="Name of the dataset"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Get the current active version of a dataset
    
    - **project_id**: ID of the project
    - **dataset_name**: Name of the dataset
    """
    # Validate project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found or you don't have access to it"
        )
    
    # Get current version
    current_version = db.query(DatasetVersion).filter(
        DatasetVersion.project_id == project_id,
        DatasetVersion.name == dataset_name,
        DatasetVersion.is_current == True
    ).first()
    
    if not current_version:
        raise HTTPException(
            status_code=404,
            detail=f"No current version found for dataset '{dataset_name}'"
        )
    
    return {
        "success": True,
        "current_version": {
            "id": str(current_version.id),
            "name": current_version.name,
            "version": current_version.version,
            "tag": current_version.tag,
            "is_current": current_version.is_current,
            "storage_path": current_version.storage_path,
            "dvc_path": current_version.dvc_path,
            "size_bytes": current_version.size_bytes,
            "num_rows": current_version.num_rows,
            "num_columns": current_version.num_columns,
            "created_at": current_version.created_at,
            "schema_info": current_version.schema_info,
            "statistics": current_version.statistics
        }
    }


@router.delete("/datasets/{project_id}/versions/{version_id}")
async def delete_dataset_version(
    project_id: str,
    version_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Delete a specific dataset version
    Note: Cannot delete the current active version
    
    - **project_id**: ID of the project
    - **version_id**: ID of the dataset version to delete
    """
    # Validate project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found or you don't have access to it"
        )
    
    # Get the version to delete
    version_to_delete = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.project_id == project_id
    ).first()
    
    if not version_to_delete:
        raise HTTPException(
            status_code=404,
            detail="Dataset version not found"
        )
    
    if version_to_delete.is_current:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the current active version. Set another version as current first."
        )
    
    # Delete the physical files if they exist
    import os
    if version_to_delete.storage_path and os.path.exists(version_to_delete.storage_path):
        try:
            if os.path.isfile(version_to_delete.storage_path):
                os.remove(version_to_delete.storage_path)
            else:
                import shutil
                shutil.rmtree(version_to_delete.storage_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete physical files: {e}")
    
    # Delete DVC file if it exists
    if version_to_delete.dvc_path and os.path.exists(version_to_delete.dvc_path):
        try:
            os.remove(version_to_delete.dvc_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete DVC file: {e}")
    
    # Delete the database record
    db.delete(version_to_delete)
    db.commit()
    
    return {
        "success": True,
        "message": f"Dataset version {version_to_delete.version} deleted successfully"
    }