"""
MongoDB Schemas using Pydantic for document validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


class ModelSessionDocument(BaseModel):
    """MongoDB document for ML training sessions"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="User who created the session")
    project_id: Optional[str] = Field(None, description="Associated project")
    
    # Model information
    model_type: Optional[str] = Field(None, description="Type of ML model")
    is_classifier: Optional[bool] = Field(None, description="Whether model is classifier")
    
    # Training data
    dataset_info: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    
    # Training results
    training_metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    cross_validation_scores: List[float] = Field(default_factory=list, description="CV scores")
    
    # Session management
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")
    
    # MLflow integration
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    mlflow_experiment_id: Optional[str] = Field(None, description="MLflow experiment ID")
    
    # Model storage
    model_storage_path: Optional[str] = Field(None, description="Path to stored model")
    dvc_model_path: Optional[str] = Field(None, description="DVC path for model")
    
    class Config:
        use_enum_values = True


class DVCMetadataDocument(BaseModel):
    """DVC metadata tracking in MongoDB"""
    file_path: str = Field(..., description="File path in DVC")
    project_id: str = Field(..., description="Associated project")
    user_id: str = Field(..., description="User who created the file")
    
    # DVC information
    dvc_file_path: str = Field(..., description="Path to .dvc file")
    md5_hash: str = Field(..., description="MD5 hash of the file")
    size: int = Field(..., description="File size in bytes")
    
    # Metadata
    file_type: str = Field(..., description="Type of file (model, dataset, etc.)")
    version: str = Field(..., description="Version identifier")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    # Storage backend information
    storage_backend: str = Field(default="local", description="Storage backend (local, s3, etc.)")
    remote_path: Optional[str] = Field(None, description="Remote storage path")
    
    # Relationships
    related_experiment_id: Optional[str] = Field(None, description="Related experiment")
    related_model_version_id: Optional[str] = Field(None, description="Related model version")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional metadata
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class MLFlowRunDocument(BaseModel):
    """MLflow run metadata in MongoDB"""
    run_id: str = Field(..., description="MLflow run ID")
    experiment_id: str = Field(..., description="MLflow experiment ID")
    
    # Project and user
    project_id: str = Field(..., description="Associated project")
    user_id: str = Field(..., description="User who created the run")
    session_id: Optional[str] = Field(None, description="Associated session")
    
    # Run details
    run_name: Optional[str] = Field(None, description="Run name")
    status: str = Field(..., description="Run status")
    start_time: datetime = Field(..., description="Run start time")
    end_time: Optional[datetime] = Field(None, description="Run end time")
    
    # Parameters and metrics
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Run parameters")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Run metrics")
    tags: Dict[str, str] = Field(default_factory=dict, description="Run tags")
    
    # Artifacts
    artifact_uri: Optional[str] = Field(None, description="Artifact storage URI")
    model_artifacts: List[str] = Field(default_factory=list, description="Model artifact paths")
    
    # DVC integration
    dvc_tracked_files: List[str] = Field(default_factory=list, description="DVC tracked files")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectConfigDocument(BaseModel):
    """Project configuration stored in MongoDB"""
    project_id: str = Field(..., description="Project identifier")
    user_id: str = Field(..., description="Project owner")
    
    # Project settings
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    
    # MLflow configuration
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(..., description="Default MLflow experiment name")
    
    # DVC configuration
    dvc_remote_name: Optional[str] = Field(None, description="DVC remote name")
    dvc_storage_config: Dict[str, Any] = Field(default_factory=dict, description="DVC storage config")
    
    # Model storage settings
    model_storage_path: str = Field(..., description="Base path for model storage")
    dataset_storage_path: str = Field(..., description="Base path for dataset storage")
    
    # Default settings
    default_hyperparameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Default hyperparameters per model type"
    )
    
    # Collaboration settings
    allowed_users: List[str] = Field(default_factory=list, description="Users with access")
    is_public: bool = Field(default=False, description="Whether project is public")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLogDocument(BaseModel):
    """Audit log for tracking all actions"""
    action_id: str = Field(..., description="Unique action identifier")
    user_id: Optional[str] = Field(None, description="User who performed action")
    project_id: Optional[str] = Field(None, description="Associated project")
    
    # Action details
    action_type: str = Field(..., description="Type of action")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of affected resource")
    
    # Change details
    old_values: Dict[str, Any] = Field(default_factory=dict, description="Previous values")
    new_values: Dict[str, Any] = Field(default_factory=dict, description="New values")
    
    # Context
    session_id: Optional[str] = Field(None, description="Session context")
    ip_address: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
