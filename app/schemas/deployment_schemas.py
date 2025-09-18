"""
Deployment configuration schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

from app.core.enums import ModelType, MetricType


class DeploymentEnvironment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStrategy(str, Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling" 
    CANARY = "canary"
    RECREATE = "recreate"


class ModelSelectionCriteria(BaseModel):
    """Criteria for automatic model selection"""
    primary_metric: MetricType = Field(..., description="Primary metric for comparison")
    metric_threshold: float = Field(..., description="Minimum threshold for primary metric")
    secondary_metrics: Dict[MetricType, float] = Field(default_factory=dict, description="Secondary metrics with thresholds")
    
    # Model requirements
    max_model_size_mb: Optional[float] = Field(None, description="Maximum model size in MB")
    max_inference_time_ms: Optional[float] = Field(None, description="Maximum inference time in milliseconds")
    
    # Selection strategy
    prefer_latest: bool = Field(True, description="Prefer newer models if metrics are equal")
    require_validation: bool = Field(True, description="Require validation before deployment")


class ContainerConfig(BaseModel):
    """Docker container configuration"""
    base_image: str = Field(default="python:3.10-slim", description="Base Docker image")
    cpu_limit: str = Field(default="1000m", description="CPU limit")
    memory_limit: str = Field(default="2Gi", description="Memory limit")
    cpu_request: str = Field(default="500m", description="CPU request")
    memory_request: str = Field(default="1Gi", description="Memory request")
    
    # Environment variables
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Health checks
    health_check_path: str = Field(default="/health", description="Health check endpoint")
    readiness_probe_path: str = Field(default="/ready", description="Readiness probe endpoint")


class DeploymentConfig(BaseModel):
    """Complete deployment configuration"""
    # Deployment metadata
    deployment_name: str = Field(..., description="Deployment name")
    project_id: str = Field(..., description="Project ID")
    user_id: str = Field(..., description="User ID")
    
    # Model selection
    model_selection: ModelSelectionCriteria = Field(..., description="Model selection criteria")
    manual_model_id: Optional[str] = Field(None, description="Manually selected model ID")
    
    # Deployment configuration
    environment: DeploymentEnvironment = Field(..., description="Target environment")
    strategy: DeploymentStrategy = Field(default=DeploymentStrategy.ROLLING, description="Deployment strategy")
    
    # Container configuration
    container: ContainerConfig = Field(default_factory=ContainerConfig, description="Container configuration")
    
    # API configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    enable_swagger: bool = Field(True, description="Enable Swagger documentation")
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    
    # Scaling configuration
    min_replicas: int = Field(default=1, description="Minimum number of replicas")
    max_replicas: int = Field(default=10, description="Maximum number of replicas")
    target_cpu_utilization: int = Field(default=70, description="Target CPU utilization for auto-scaling")
    
    # Networking
    service_port: int = Field(default=8000, description="Service port")
    ingress_enabled: bool = Field(False, description="Enable ingress")
    domain: Optional[str] = Field(None, description="Custom domain")
    
    # Security
    enable_authentication: bool = Field(True, description="Enable API authentication")
    allowed_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")
    
    # Monitoring and logging
    log_level: str = Field(default="INFO", description="Log level")
    enable_tracing: bool = Field(False, description="Enable distributed tracing")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "protected_namespaces": ()
    }


class ModelComparison(BaseModel):
    """Model comparison result"""
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    
    # Metrics
    metrics: Dict[str, float] = Field(..., description="Model metrics")
    score: float = Field(..., description="Computed comparison score")
    
    # Metadata
    created_at: datetime = Field(..., description="Model creation time")
    size_mb: float = Field(..., description="Model size in MB")
    training_time_minutes: Optional[float] = Field(None, description="Training time in minutes")
    
    # Status
    is_eligible: bool = Field(..., description="Whether model meets criteria")
    eligibility_reasons: List[str] = Field(default_factory=list, description="Reasons for eligibility status")
    model_config = {
        "protected_namespaces": ()
    }

class DeploymentRequest(BaseModel):
    """API request for model deployment"""
    project_id: str = Field(..., description="Project ID")
    deployment_config: DeploymentConfig = Field(..., description="Deployment configuration")
    
    # Optional manual override
    force_deploy: bool = Field(default=False, description="Force deployment without validation")
    dry_run: bool = Field(default=False, description="Perform dry run without actual deployment")
    async_deployment: bool = Field(default=False, description="Whether to run deployment asynchronously in background")


class DeploymentResponse(BaseModel):
    """API response for deployment"""
    deployment_id: str = Field(..., description="Deployment ID")
    status: str = Field(..., description="Deployment status")
    message: str = Field(..., description="Status message")
    
    # Selected model info
    selected_model: Optional[ModelComparison] = Field(None, description="Selected model information")
    model_comparisons: List[ModelComparison] = Field(default_factory=list, description="All model comparisons")
    
    # Deployment details
    api_endpoint: Optional[str] = Field(None, description="Deployed API endpoint")
    swagger_url: Optional[str] = Field(None, description="Swagger documentation URL")
    
    # Container info
    container_image: Optional[str] = Field(None, description="Container image name")
    kubernetes_namespace: Optional[str] = Field(None, description="Kubernetes namespace")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_ready_time: Optional[datetime] = Field(None, description="Estimated ready time")
    model_config = {
        "protected_namespaces": ()
    }

class DeploymentStatus(BaseModel):
    """Deployment status information"""
    deployment_id: str = Field(..., description="Deployment ID")
    status: str = Field(..., description="Current status")
    
    # Health information
    health_status: str = Field(..., description="Health check status")
    ready_replicas: int = Field(default=0, description="Number of ready replicas")
    total_replicas: int = Field(default=0, description="Total number of replicas")
    
    # Performance metrics
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    request_count: Optional[int] = Field(None, description="Total request count")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = Field(None, description="Last health check time")
