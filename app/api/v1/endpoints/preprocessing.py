"""
Data preprocessing endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session
import tempfile
import pandas as pd
from pathlib import Path

from app.services.preprocessing import PreprocessingService
from app.services.file_service import FileService
from app.schemas.schemas import PreprocessingRequest, PreprocessingResponse, ErrorResponse
from app.core.database import get_session
from app.core.auth import get_current_active_user
from app.models.sql_models import User, Project, DatasetVersion


def to_serializable(val):
    """Convert numpy types to JSON serializable Python types"""
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    elif isinstance(val, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(val)
    elif isinstance(val, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, np.dtype):
        return str(val)
    return val

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


def create_preprocessed_dataset_version(db: Session, project_id: str, dataset_name: str, 
                                       processed_data: pd.DataFrame, original_file_path: str) -> DatasetVersion:
    """
    Create a new dataset version record for preprocessed data
    """
    version = get_next_dataset_version(db, project_id, dataset_name)
    
    # Save processed data to file
    file_extension = Path(original_file_path).suffix
    processed_filename = f"preprocessed_{dataset_name}_{version}{file_extension}"
    processed_file_path = Path(original_file_path).parent / processed_filename
    
    # Save processed data
    if file_extension.lower() == '.csv':
        processed_data.to_csv(processed_file_path, index=False)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        processed_data.to_excel(processed_file_path, index=False)
    elif file_extension.lower() == '.parquet':
        processed_data.to_parquet(processed_file_path, index=False)
    elif file_extension.lower() == '.json':
        processed_data.to_json(processed_file_path, orient='records', indent=2)
    
    # Create dataset version record
    dataset_version = DatasetVersion(
        project_id=project_id,
        name=dataset_name,
        version=version,
        tag="preprocessed data",
        storage_path=str(processed_file_path),
        size_bytes=processed_file_path.stat().st_size if processed_file_path.exists() else 0,
        num_rows=len(processed_data),
        num_columns=len(processed_data.columns),
        schema_info={col: str(dtype) for col, dtype in processed_data.dtypes.items()},
        statistics={
            'mean': processed_data.select_dtypes(include=[np.number]).mean().to_dict(),
            'std': processed_data.select_dtypes(include=[np.number]).std().to_dict(),
            'null_counts': processed_data.isnull().sum().to_dict()
        }
    )
    
    db.add(dataset_version)
    db.commit()
    db.refresh(dataset_version)
    
    return dataset_version


async def get_data_dependency(file_id: str, project_id: str = None):
    """Dependency to load data from file"""
    result = await file_service.load_data(file_id, project_id=project_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    return result["data"]


@router.post("/apply", response_model=PreprocessingResponse)
async def apply_preprocessing(
    request: PreprocessingRequest,
    project_id: str = Query(..., description="ID of the project"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Apply preprocessing operations to data and create new dataset version
    Creates V2 from V1, or V2.1, V2.2, etc. for subsequent preprocessing
    
    - **file_id**: ID of the uploaded data file
    - **operations**: Dictionary mapping operation types to columns
    - **is_categorical**: Whether to treat columns as categorical
    - **project_id**: Project ID for versioning
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
    
    # Load data with project context
    data = await get_data_dependency(request.file_id, project_id=project_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Apply preprocessing
    result = preprocess_service.apply_preprocessing(
        choices=request.operations,
        is_categorical=request.is_categorical
    )
    
    # Get processed data summary
    data_summary = preprocess_service.get_data_summary()
    
    # Get the processed data
    processed_data = preprocess_service.data
    
    # Create new dataset version for preprocessed data
    try:
        # Find original dataset version to get dataset name and file path
        original_dataset = db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id
        ).order_by(DatasetVersion.created_at.desc()).first()
        
        if original_dataset:
            dataset_name = original_dataset.name
            original_file_path = original_dataset.storage_path
        else:
            # Fallback if no original dataset found
            dataset_name = f"dataset_{request.file_id}"
            original_file_path = f"uploads/{project_id}/{request.file_id}"
        
        # Create new version for preprocessed data
        preprocessed_version = create_preprocessed_dataset_version(
            db=db,
            project_id=project_id,
            dataset_name=dataset_name,
            processed_data=processed_data,
            original_file_path=original_file_path
        )
        
        # Add version info to result
        result["dataset_version"] = preprocessed_version.version
        result["dataset_tag"] = preprocessed_version.tag
        result["dataset_version_id"] = str(preprocessed_version.id)
        result["preprocessed_file_path"] = preprocessed_version.storage_path
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to create preprocessed dataset version: {e}")
        # Don't fail the preprocessing if versioning fails
    
    # Convert numpy types to JSON serializable
    serialized_result = to_serializable(result)
    serialized_summary = to_serializable(data_summary)
    
    return PreprocessingResponse(
        success=True,
        applied_operations=serialized_result["applied_operations"],
        errors=serialized_result["errors"],
        data_shape_before=serialized_result["data_shape_before"],
        data_shape_after=serialized_result["data_shape_after"],
        columns_before=serialized_result["columns_before"],
        columns_after=serialized_result["columns_after"],
        data_summary=serialized_summary,
        message=f"Preprocessing applied successfully. Created {result.get('dataset_version', 'new version')} ({result.get('dataset_tag', 'preprocessed data')})"
    )


@router.post("/delete-columns")
async def delete_columns(
    file_id: str,
    columns: List[str],
    project_id: str = Query(..., description="ID of the project")
):
    """
    Delete specified columns from the dataset
    
    - **file_id**: ID of the uploaded data file
    - **columns**: List of column names to delete
    - **project_id**: Project ID to locate the file
    """
    # Load data
    data = await get_data_dependency(file_id, project_id=project_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Delete columns
    result = preprocess_service.delete_columns(columns)
    
    return {
        "success": True,
        "deleted_columns": result["deleted_columns"],
        "not_found_columns": result["not_found_columns"],
        "shape_before": result["shape_before"],
        "shape_after": result["shape_after"]
    }


@router.get("/data-summary/{file_id}")
async def get_data_summary(
    file_id: str,
    project_id: str = Query(..., description="ID of the project")
):
    """
    Get comprehensive data summary for preprocessing planning
    
    - **file_id**: ID of the uploaded data file
    - **project_id**: Project ID to locate the file
    """
    # Load data
    data = await get_data_dependency(file_id, project_id=project_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Get data summary
    summary = preprocess_service.get_data_summary()
    
    # Convert numpy types to JSON serializable
    serialized_summary = to_serializable(summary)
    
    return {
        "success": True,
        "data_summary": serialized_summary
    }


@router.get("/dataset-versions/{project_id}")
async def get_dataset_versions(
    project_id: str,
    dataset_name: Optional[str] = Query(None, description="Filter by dataset name"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """
    Get all dataset versions for a project
    
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
    
    # Get all versions ordered by creation date
    versions = query.order_by(DatasetVersion.created_at.desc()).all()
    
    # Format response
    version_list = []
    for version in versions:
        version_list.append({
            "id": str(version.id),
            "name": version.name,
            "version": version.version,
            "tag": version.tag,
            "storage_path": version.storage_path,
            "size_bytes": version.size_bytes,
            "num_rows": version.num_rows,
            "num_columns": version.num_columns,
            "created_at": version.created_at.isoformat() if version.created_at else None,
            "schema_info": version.schema_info,
            "statistics": version.statistics
        })
    
    return {
        "success": True,
        "project_id": project_id,
        "dataset_name": dataset_name,
        "versions": version_list,
        "total_versions": len(version_list)
    }


@router.get("/recommendations/{file_id}")
async def get_preprocessing_recommendations(
    file_id: str,
    project_id: str = Query(..., description="ID of the project")
):
    """
    Get automatic preprocessing recommendations based on data analysis
    
    - **file_id**: ID of the uploaded data file
    - **project_id**: Project ID to locate the file
    """
    # Load data
    data = await get_data_dependency(file_id, project_id=project_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Get data summary for analysis
    summary = preprocess_service.get_data_summary()
    
    recommendations = []
    
    # Recommend imputation for columns with missing values
    for column, missing_count in summary["missing_values"].items():
        if missing_count > 0:
            missing_percentage = (missing_count / summary["shape"][0]) * 100
            
            if column in summary["numeric_columns"]:
                if missing_percentage < 5:
                    recommendations.append({
                        "operation": "imputation",
                        "column": column,
                        "method": "mean",
                        "reason": f"Column has {missing_percentage:.1f}% missing values - recommend mean imputation"
                    })
                elif missing_percentage < 20:
                    recommendations.append({
                        "operation": "imputation",
                        "column": column,
                        "method": "knn",
                        "reason": f"Column has {missing_percentage:.1f}% missing values - recommend KNN imputation"
                    })
                else:
                    recommendations.append({
                        "operation": "deletion",
                        "column": column,
                        "method": "drop_column",
                        "reason": f"Column has {missing_percentage:.1f}% missing values - consider dropping"
                    })
            else:
                recommendations.append({
                    "operation": "imputation",
                    "column": column,
                    "method": "mode",
                    "reason": f"Categorical column has {missing_percentage:.1f}% missing values - recommend mode imputation"
                })
    
    # Recommend encoding for categorical columns
    for column in summary["categorical_columns"]:
        unique_values = data[column].nunique()
        if unique_values > 10:
            recommendations.append({
                "operation": "encoding",
                "column": column,
                "method": "label",
                "reason": f"High cardinality categorical column ({unique_values} unique values) - recommend label encoding"
            })
        else:
            recommendations.append({
                "operation": "encoding",
                "column": column,
                "method": "one-hot",
                "reason": f"Low cardinality categorical column ({unique_values} unique values) - recommend one-hot encoding"
            })
    
    # Recommend normalization for numeric columns with different scales
    numeric_data = data[summary["numeric_columns"]]
    if len(summary["numeric_columns"]) > 1:
        ranges = numeric_data.max() - numeric_data.min()
        max_range = ranges.max()
        min_range = ranges.min()
        
        if max_range / min_range > 10:  # Large scale differences
            for column in summary["numeric_columns"]:
                recommendations.append({
                    "operation": "normalization",
                    "column": column,
                    "method": "standard",
                    "reason": "Different scales detected across numeric columns - recommend standardization"
                })
    
    # Recommend outlier removal for numeric columns
    for column in summary["numeric_columns"]:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))).sum()
        
        if outlier_count > 0:
            outlier_percentage = (outlier_count / len(data)) * 100
            if outlier_percentage > 5:
                recommendations.append({
                    "operation": "cleaning",
                    "column": column,
                    "method": "outlier_removal",
                    "reason": f"Column has {outlier_percentage:.1f}% outliers - recommend cleaning"
                })
    
    # Convert numpy types to JSON serializable
    serialized_summary = to_serializable(summary)
    serialized_recommendations = to_serializable(recommendations)
    
    return {
        "success": True,
        "recommendations": serialized_recommendations,
        "total_recommendations": len(recommendations),
        "data_summary": serialized_summary
    }


from pydantic import BaseModel

class ExportRequest(BaseModel):
    file_id: str
    operations: Dict[str, str]
    export_format: str = "csv"
    filename: Optional[str] = None

@router.post("/export")
async def export_processed_data(
    request: ExportRequest,
    project_id: str = Query(..., description="ID of the project")
):
    """
    Apply preprocessing and export the processed data
    
    - **file_id**: ID of the uploaded data file
    - **operations**: Dictionary mapping operation types to columns
    - **export_format**: Export format (csv, excel, json, parquet)
    - **filename**: Optional custom filename
    - **project_id**: Project ID to locate the file
    """
    # Load data
    data = await get_data_dependency(request.file_id, project_id=project_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Apply preprocessing
    preprocess_result = preprocess_service.apply_preprocessing(request.operations)
    
    if preprocess_result["errors"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Preprocessing errors: {'; '.join(preprocess_result['errors'])}"
        )
    
    # Get processed data
    processed_data = preprocess_service.get_processed_data()
    
    # Export data
    export_filename = request.filename or f"processed_data_{request.file_id}"
    export_result = file_service.export_data(processed_data, export_filename, request.export_format)
    
    if not export_result.get("success", False):
        raise HTTPException(status_code=500, detail=export_result.get("error", "Export failed"))
    
    return {
        "success": True,
        "preprocessing_applied": preprocess_result["applied_operations"],
        "export_file_id": export_result["file_id"],
        "export_format": request.export_format,
        "file_size": export_result["size"],
        "processed_shape": processed_data.shape
    }
