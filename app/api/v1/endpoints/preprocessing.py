"""
Data preprocessing endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import numpy as np

from app.services.preprocessing import PreprocessingService
from app.services.file_service import FileService
from app.schemas.schemas import PreprocessingRequest, PreprocessingResponse, ErrorResponse


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


async def get_data_dependency(file_id: str):
    """Dependency to load data from file"""
    result = await file_service.load_data(file_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    return result["data"]


@router.post("/apply", response_model=PreprocessingResponse)
async def apply_preprocessing(request: PreprocessingRequest):
    """
    Apply preprocessing operations to data
    
    - **file_id**: ID of the uploaded data file
    - **operations**: Dictionary mapping operation types to columns
    - **is_categorical**: Whether to treat columns as categorical
    """
    # Load data
    data = await get_data_dependency(request.file_id)
    
    # Create preprocessing service
    preprocess_service = PreprocessingService(data)
    
    # Apply preprocessing
    result = preprocess_service.apply_preprocessing(
        choices=request.operations,
        is_categorical=request.is_categorical
    )
    
    # Get processed data summary
    data_summary = preprocess_service.get_data_summary()
    
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
        data_summary=serialized_summary
    )


@router.post("/delete-columns")
async def delete_columns(
    file_id: str,
    columns: List[str]
):
    """
    Delete specified columns from the dataset
    
    - **file_id**: ID of the uploaded data file
    - **columns**: List of column names to delete
    """
    # Load data
    data = await get_data_dependency(file_id)
    
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
async def get_data_summary(file_id: str):
    """
    Get comprehensive data summary for preprocessing planning
    
    - **file_id**: ID of the uploaded data file
    """
    # Load data
    data = await get_data_dependency(file_id)
    
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


@router.get("/recommendations/{file_id}")
async def get_preprocessing_recommendations(file_id: str):
    """
    Get automatic preprocessing recommendations based on data analysis
    
    - **file_id**: ID of the uploaded data file
    """
    # Load data
    data = await get_data_dependency(file_id)
    
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
async def export_processed_data(request: ExportRequest):
    """
    Apply preprocessing and export the processed data
    
    - **file_id**: ID of the uploaded data file
    - **operations**: Dictionary mapping operation types to columns
    - **export_format**: Export format (csv, excel, json, parquet)
    - **filename**: Optional custom filename
    """
    # Load data
    data = await get_data_dependency(request.file_id)
    
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
