"""
File management service for handling uploads and data processing
"""
import os
import pandas as pd
import numpy as np
from fastapi import UploadFile
import aiofiles
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path
import json

from app.core.config import get_settings


class FileService:
    """Service for file upload and management operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Supported file types
        self.supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    async def upload_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Upload and validate a file
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Dictionary with upload results and file info
        """
        try:
            # Validate file
            validation_result = self._validate_file(file)
            if not validation_result["valid"]:
                return validation_result
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix.lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file info
            file_info = await self._get_file_info(file_path, file.filename)
            
            return {
                "success": True,
                "file_id": unique_filename,
                "file_path": str(file_path),
                "original_filename": file.filename,
                "file_info": file_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def load_data(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load data from uploaded file
        
        Args:
            file_id: Unique file identifier
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            Dictionary with loaded data and metadata
        """
        try:
            file_path = self.upload_dir / file_id
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File {file_id} not found"
                }
            
            # Determine file type and load accordingly
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                data = pd.read_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                data = pd.read_parquet(file_path, **kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}"
                }
            
            # Get data summary
            data_summary = self._get_data_summary(data)
            
            return {
                "success": True,
                "data": data,
                "data_summary": data_summary,
                "file_id": file_id,
                "shape": data.shape
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id
            }
    
    async def get_data_preview(self, file_id: str, n_rows: int = 5) -> Dict[str, Any]:
        """
        Get a preview of the data without loading the entire file
        
        Args:
            file_id: Unique file identifier
            n_rows: Number of rows to preview
            
        Returns:
            Dictionary with data preview and basic info
        """
        try:
            file_path = self.upload_dir / file_id
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File {file_id} not found"
                }
            
            file_extension = file_path.suffix.lower()
            
            # Load limited data for preview
            if file_extension == '.csv':
                data = pd.read_csv(file_path, nrows=n_rows)
                total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, nrows=n_rows)
                # For Excel, we need to load to count rows (less efficient)
                full_data = pd.read_excel(file_path)
                total_rows = len(full_data)
            elif file_extension == '.json':
                data = pd.read_json(file_path)
                total_rows = len(data)
                data = data.head(n_rows)
            elif file_extension == '.parquet':
                data = pd.read_parquet(file_path)
                total_rows = len(data)
                data = data.head(n_rows)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}"
                }
            
            # Convert to dict for JSON serialization
            preview_data = data.to_dict(orient='records')
            
            return {
                "success": True,
                "preview": preview_data,
                "columns": list(data.columns),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "shape": (total_rows, len(data.columns)),
                "preview_rows": len(preview_data),
                "file_id": file_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id
            }
    
    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Delete an uploaded file
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Dictionary with deletion result
        """
        try:
            file_path = self.upload_dir / file_id
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File {file_id} not found"
                }
            
            file_path.unlink()
            
            return {
                "success": True,
                "message": f"File {file_id} deleted successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_files(self) -> Dict[str, Any]:
        """
        List all uploaded files
        
        Returns:
            Dictionary with list of files and their info
        """
        try:
            files = []
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    file_info = {
                        "file_id": file_path.name,
                        "original_name": file_path.name,  # Could be enhanced to store original names
                        "size": file_path.stat().st_size,
                        "created": file_path.stat().st_ctime,
                        "extension": file_path.suffix.lower()
                    }
                    files.append(file_info)
            
            return {
                "success": True,
                "files": files,
                "total_files": len(files)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        # Check filename
        if not file.filename:
            return {
                "valid": False,
                "error": "No filename provided"
            }
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.supported_extensions:
            return {
                "valid": False,
                "error": f"Unsupported file type. Supported types: {', '.join(self.supported_extensions)}"
            }
        
        # Check file size (this is approximate for FastAPI UploadFile)
        if hasattr(file, 'size') and file.size > self.max_file_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {self.max_file_size / (1024*1024):.1f}MB"
            }
        
        return {"valid": True}
    
    async def _get_file_info(self, file_path: Path, original_filename: str) -> Dict[str, Any]:
        """Get detailed file information"""
        file_stat = file_path.stat()
        
        return {
            "size": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024*1024), 2),
            "created": file_stat.st_ctime,
            "extension": file_path.suffix.lower(),
            "original_filename": original_filename
        }
    
    def _get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": data.isnull().sum().to_dict(),
            "memory_usage_mb": round(data.memory_usage(deep=True).sum() / (1024*1024), 2)
        }
        
        # Add basic statistics for numeric columns
        if numeric_columns:
            numeric_stats = data[numeric_columns].describe().to_dict()
            summary["numeric_statistics"] = numeric_stats
        
        # Add value counts for categorical columns (top 5 values)
        if categorical_columns:
            categorical_info = {}
            for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                value_counts = data[col].value_counts().head(5).to_dict()
                categorical_info[col] = {
                    "unique_values": int(data[col].nunique()),
                    "top_values": value_counts
                }
            summary["categorical_info"] = categorical_info
        
        return summary
    
    def export_data(self, data: pd.DataFrame, filename: str, format: str = 'csv') -> Dict[str, Any]:
        """
        Export processed data to file
        
        Args:
            data: DataFrame to export
            filename: Name for the exported file
            format: Export format ('csv', 'excel', 'json', 'parquet')
            
        Returns:
            Dictionary with export results
        """
        try:
            # Generate unique filename
            file_id = f"{uuid.uuid4()}_{filename}"
            
            if format == 'csv':
                file_path = self.upload_dir / f"{file_id}.csv"
                data.to_csv(file_path, index=False)
            elif format == 'excel':
                file_path = self.upload_dir / f"{file_id}.xlsx"
                data.to_excel(file_path, index=False)
            elif format == 'json':
                file_path = self.upload_dir / f"{file_id}.json"
                data.to_json(file_path, orient='records', indent=2)
            elif format == 'parquet':
                file_path = self.upload_dir / f"{file_id}.parquet"
                data.to_parquet(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported export format: {format}"
                }
            
            return {
                "success": True,
                "file_id": file_path.name,
                "file_path": str(file_path),
                "format": format,
                "size": file_path.stat().st_size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
