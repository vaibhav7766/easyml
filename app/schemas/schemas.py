"""
Pydantic schemas for request/response models
"""
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

from app.core.enums import PlotType, PreprocessingOption, ModelType, ProjectStatus, TaskType


class BaseSchema(BaseModel):
    """Base schema with common configurations"""
    class Config:
        from_attributes = True
        use_enum_values = True


# Project Schemas
class ProjectCreate(BaseSchema):
    """Schema for creating a new project"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    user_id: str = Field(..., min_length=1)


class ProjectResponse(BaseSchema):
    """Schema for project response"""
    id: str
    name: str
    description: Optional[str] = None
    owner_id: str  # Match the actual Project model field
    status: str = "ongoing"
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @model_validator(mode='before')
    @classmethod
    def convert_uuids(cls, data):
        """Convert UUID objects to strings"""
        import uuid
        if hasattr(data, '__dict__'):
            # Handle SQLAlchemy model objects
            data_dict = {}
            for key, value in data.__dict__.items():
                if isinstance(value, uuid.UUID):
                    data_dict[key] = str(value)
                else:
                    data_dict[key] = value
            return data_dict
        elif isinstance(data, dict):
            # Handle dictionary input
            for key, value in data.items():
                if isinstance(value, uuid.UUID):
                    data[key] = str(value)
            return data
        return data


# Upload Schemas
class FileInfoResponse(BaseSchema):
    """Schema for file info response"""
    success: bool
    file_id: str
    original_filename: str
    file_info: Dict[str, Any]
    message: str
    project_id: Optional[str] = None


class DataPreviewResponse(BaseSchema):
    """Schema for data preview response"""
    success: bool
    preview: List[Dict[str, Any]]
    columns: List[str]
    dtypes: Dict[str, str]
    shape: tuple
    preview_rows: int


class ErrorResponse(BaseSchema):
    """Schema for error responses"""
    success: bool = False
    error: str
    detail: Optional[str] = None


class FileUploadResponse(BaseSchema):
    """Schema for file upload response"""
    message: str
    project_id: str
    filename: str
    file_size: int
    columns: List[str]
    rows: int


# Preprocessing Schemas
class PreprocessingRequest(BaseSchema):
    """Schema for preprocessing request"""
    file_id: str
    operations: Dict[str, Union[str, List[str]]] = Field(..., description="Preprocessing operations mapping")
    is_categorical: bool = False


class PreprocessingResponse(BaseSchema):
    """Schema for preprocessing response"""
    success: bool
    applied_operations: List[str]
    errors: List[str]
    data_shape_before: tuple
    data_shape_after: tuple
    columns_before: List[str]
    columns_after: List[str]
    data_summary: Dict[str, Any]
    message: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_tag: Optional[str] = None


class DatasetVersionResponse(BaseSchema):
    """Schema for dataset version information"""
    id: str
    name: str
    version: str
    tag: str
    storage_path: str
    size_bytes: Optional[int] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    created_at: Optional[datetime] = None
    schema_info: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None


class DatasetVersionListResponse(BaseSchema):
    """Schema for dataset version list response"""
    success: bool
    project_id: str
    dataset_name: Optional[str] = None
    versions: List[DatasetVersionResponse]
    total_versions: int


# Visualization Schemas
class VisualizationRequest(BaseSchema):
    """Schema for visualization request"""
    file_id: str
    plot_type: PlotType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    plot_params: Dict[str, Any] = Field(default_factory=dict)


class VisualizationResponse(BaseSchema):
    """Schema for visualization response"""
    success: bool
    plotly_json: str = Field(..., description="Plotly JSON for interactive charts")
    plot_type: str
    columns_used: Dict[str, Optional[str]]
    plot_info: Dict[str, Any]


# Model Training Schemas
class ModelTrainingRequest(BaseSchema):
    """Schema for model training request"""
    target_column: str
    model_type: ModelType
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    is_categorical: bool = False
    hyperparameters: Optional[Dict[str, Any]] = None
    use_cross_validation: bool = True
    cv_folds: int = Field(1, ge=1, le=10)
    session_id: str = "default"


class ModelTrainingResponse(BaseSchema):
    """Schema for model training response"""
    success: bool
    model_type: str
    is_classifier: bool
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Dict[str, Any]
    data_shape: tuple
    session_id: str


class PredictionRequest(BaseSchema):
    """Schema for prediction request"""
    file_id: str
    session_id: str


class PredictionResponse(BaseSchema):
    """Schema for prediction response"""
    success: bool
    predictions: List[Any]
    model_type: Optional[str] = None
    is_classifier: bool = False
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[str]] = None
    session_id: str


# Feature Selection Schemas (Legacy - integrated into preprocessing)
class FeatureSelectionRequest(BaseSchema):
    """Schema for feature selection request"""
    file_id: str
    method: str = Field(..., description="Feature selection method")
    target_column: str
    n_features: Optional[int] = Field(None, ge=1)


class FeatureSelectionResponse(BaseSchema):
    """Schema for feature selection response"""
    success: bool
    selected_features: List[str]
    feature_scores: Dict[str, float]
    method_used: str


class HyperparameterTuningRequest(BaseSchema):
    """Schema for hyperparameter tuning request"""
    file_id: str
    target_column: str
    model_type: ModelType
    param_grid: Dict[str, List[Any]]
    cv_folds: int = Field(5, ge=3, le=10)
    scoring: Optional[str] = None
    session_id: str = "default"
    preprocessing_operations: Optional[Dict[str, str]] = None
    is_categorical: bool = False


class HyperparameterTuningResponse(BaseSchema):
    """Schema for hyperparameter tuning response"""
    success: bool
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, Any]
    model_type: str
    scoring: str
    cv_folds: int
    session_id: str


# Dashboard Schemas (Legacy - use individual endpoints)
class DashboardResponse(BaseSchema):
    """Schema for dashboard response"""
    user_id: str
    total_projects: int
    active_projects: int
    completed_projects: int
    recent_projects: List[ProjectResponse]
    system_stats: Dict[str, Any]
