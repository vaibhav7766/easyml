"""
File upload endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional

from app.services.file_service import FileService
from app.schemas.schemas import FileInfoResponse, DataPreviewResponse, ErrorResponse

router = APIRouter()
file_service = FileService()


@router.post("/", response_model=FileInfoResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file (CSV, Excel, JSON, Parquet)
    
    - **file**: The data file to upload
    """
    result = await file_service.upload_file(file)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
    
    return FileInfoResponse(
        success=True,
        file_id=result["file_id"],
        original_filename=result["original_filename"],
        file_info=result["file_info"],
        message="File uploaded successfully"
    )


@router.get("/preview/{file_id}", response_model=DataPreviewResponse)
async def get_file_preview(
    file_id: str,
    n_rows: int = Query(5, ge=1, le=100, description="Number of rows to preview")
):
    """
    Get a preview of uploaded data file
    
    - **file_id**: Unique identifier of the uploaded file
    - **n_rows**: Number of rows to preview (1-100)
    """
    result = await file_service.get_data_preview(file_id, n_rows)
    
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
async def list_files():
    """
    List all uploaded files
    """
    result = file_service.list_files()
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to list files"))
    
    return {
        "success": True,
        "files": result["files"],
        "total_files": result["total_files"]
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file
    
    - **file_id**: Unique identifier of the file to delete
    """
    result = file_service.delete_file(file_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    
    return {
        "success": True,
        "message": result["message"]
    }


@router.get("/info/{file_id}")
async def get_file_info(file_id: str):
    """
    Get detailed information about an uploaded file
    
    - **file_id**: Unique identifier of the file
    """
    result = await file_service.load_data(file_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    
    return {
        "success": True,
        "file_id": file_id,
        "data_summary": result["data_summary"],
        "shape": result["shape"]
    }
