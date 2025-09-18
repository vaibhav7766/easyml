"""
Model Deployment API Endpoints
Provides endpoints for automated model deployment with Docker isolation and comparison
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio

from app.core.database import get_session
from app.core.auth import get_current_user
from app.models.sql_models import User, Project, ModelDeployment
from app.schemas.deployment_schemas import (
    DeploymentRequest, DeploymentResponse, DeploymentStatus,
    DeploymentConfig, ModelComparison, ContainerConfig,
    ModelSelectionCriteria, DeploymentEnvironment
)
from app.core.enums import MetricType
from app.services.model_deployment_service import ModelDeploymentService

router = APIRouter(prefix="/deployments", tags=["Model Deployment"])
deployment_service = ModelDeploymentService()


@router.post("/deploy", response_model=DeploymentResponse)
async def deploy_model(
    request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Deploy a model with automatic comparison and selection
    
    This endpoint:
    1. Compares all models in the project by evaluation metrics
    2. Selects the best model (or allows manual selection)
    3. Creates isolated Docker container with model API
    4. Deploys to Kubernetes with auto-scaling
    5. Sets up GitHub automation for CI/CD
    """
    
    # Verify project access
    project = db.query(Project).filter(
        Project.id == request.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or access denied"
        )
    
    # Start deployment process
    if request.async_deployment:
        # Run deployment in background
        background_tasks.add_task(
            deployment_service.deploy_model,
            request, db
        )
        
        return DeploymentResponse(
            deployment_id="pending",
            status="queued",
            message="Deployment queued for background processing"
        )
    else:
        # Synchronous deployment
        return await deployment_service.deploy_model(request, db)


@router.post("/compare-models/{project_id}", response_model=List[ModelComparison])
async def compare_project_models(
    project_id: str,
    criteria: ModelSelectionCriteria,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Compare all models in a project by evaluation metrics
    
    Returns ranked list of models with scores and eligibility for deployment
    """
    
    # Verify project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or access denied"
        )
    
    # Get models and compare
    models = await deployment_service._get_project_models(project_id, db)
    
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No models found in this project"
        )
    
    comparisons = await deployment_service._compare_models(models, criteria)
    return comparisons


@router.get("/status/{deployment_id}", response_model=DeploymentStatus)
async def get_deployment_status(
    deployment_id: str,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get current status of a deployment
    
    Returns health status, replica counts, and deployment progress
    """
    
    
    # Verify deployment access
    deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail="Deployment not found"
        )
    
    # Check project access
    project = db.query(Project).filter(
        Project.id == deployment.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )
    
    status = await deployment_service.get_deployment_status(deployment_id, db)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail="Deployment status not available"
        )
    
    return status


@router.get("/list/{project_id}", response_model=List[DeploymentResponse])
async def list_project_deployments(
    project_id: str,
    environment: Optional[DeploymentEnvironment] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    List all deployments for a project
    
    Optional filters by environment and status
    """
    
    # Verify project access
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found or access denied"
        )
    
    # Build query
    query = db.query(ModelDeployment).filter(
        ModelDeployment.project_id == project_id
    )
    
    if environment:
        query = query.filter(ModelDeployment.environment == environment.value)
    
    if status:
        query = query.filter(ModelDeployment.status == status)
    
    deployments = query.order_by(ModelDeployment.created_at.desc()).all()
    
    # Convert to response format
    response_list = []
    for deployment in deployments:
        response_list.append(
            DeploymentResponse(
                deployment_id=deployment.id,
                status=deployment.status,
                api_endpoint=deployment.endpoint_url,
                container_image=deployment.container_image,
                kubernetes_namespace=deployment.kubernetes_namespace,
                message=f"Deployment in {deployment.environment} environment"
            )
        )
    
    return response_list


@router.post("/stop/{deployment_id}")
async def stop_deployment(
    deployment_id: str,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Stop and remove a deployment
    
    Scales down Kubernetes deployment and marks as inactive
    """
    
    # Verify deployment access
    deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail="Deployment not found"
        )
    
    # Check project access
    project = db.query(Project).filter(
        Project.id == deployment.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )
    
    try:
        # Scale down deployment
        import subprocess
        subprocess.run([
            "kubectl", "scale", "deployment", f"model-{deployment_id}",
            "--replicas=0", "-n", deployment.kubernetes_namespace
        ], check=True)
        
        # Update status
        deployment.status = "stopped"
        deployment.ready_replicas = 0
        db.commit()
        
        return {"message": "Deployment stopped successfully"}
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop deployment: {str(e)}"
        )


@router.post("/scale/{deployment_id}")
async def scale_deployment(
    deployment_id: str,
    replicas: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Scale a deployment to specified number of replicas
    """
    
    if replicas < 0 or replicas > 10:
        raise HTTPException(
            status_code=400,
            detail="Replicas must be between 0 and 10"
        )
    
    # Verify deployment access
    deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail="Deployment not found"
        )
    
    # Check project access
    project = db.query(Project).filter(
        Project.id == deployment.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )
    
    try:
        # Scale deployment
        import subprocess
        subprocess.run([
            "kubectl", "scale", "deployment", f"model-{deployment_id}",
            f"--replicas={replicas}", "-n", deployment.kubernetes_namespace
        ], check=True)
        
        # Update records
        deployment.total_replicas = replicas
        if replicas == 0:
            deployment.status = "stopped"
        else:
            deployment.status = "active"
        
        db.commit()
        
        return {"message": f"Deployment scaled to {replicas} replicas"}
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scale deployment: {str(e)}"
        )


@router.get("/config-template", response_model=DeploymentConfig)
async def get_deployment_config_template():
    """
    Get a template deployment configuration with default values
    
    Useful for understanding available configuration options
    """
    
    return DeploymentConfig(
        deployment_name="example-model-deployment",
        project_id="your-project-id",
        user_id="your-user-id",
        model_selection=ModelSelectionCriteria(
            primary_metric=MetricType.ACCURACY,
            metric_threshold=0.8,
            secondary_metrics={
                MetricType.PRECISION: 0.7,
                MetricType.RECALL: 0.7
            },
            prefer_latest=True,
            max_model_size_mb=100
        ),
        environment=DeploymentEnvironment.STAGING,
        container=ContainerConfig(
            base_image="python:3.10-slim",
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="128Mi",
            memory_limit="512Mi",
            env_vars={
                "LOG_LEVEL": "INFO",
                "WORKERS": "1"
            }
        ),
        service_port=8000,
        api_prefix="/api/v1",
        min_replicas=1,
        max_replicas=3,
        target_cpu_utilization=70,
        enable_swagger=True,
        auto_scaling=True,
        log_level="INFO"
    )


@router.post("/validate-config")
async def validate_deployment_config(config: DeploymentConfig):
    """
    Validate a deployment configuration without actually deploying
    
    Returns validation results and estimated resource requirements
    """
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "estimated_resources": {}
    }
    
    # Validate resource requests
    try:
        cpu_request = float(config.container.cpu_request.replace('m', '')) / 1000
        memory_request = int(config.container.memory_request.replace('Mi', ''))
        
        validation_results["estimated_resources"] = {
            "cpu_cores": cpu_request * config.max_replicas,
            "memory_mb": memory_request * config.max_replicas,
            "estimated_cost_per_hour": (cpu_request * 0.1 + memory_request * 0.001) * config.max_replicas
        }
        
    except ValueError:
        validation_results["valid"] = False
        validation_results["errors"].append("Invalid CPU or memory format")
    
    # Validate metric thresholds
    if config.model_selection.metric_threshold < 0 or config.model_selection.metric_threshold > 1:
        validation_results["warnings"].append("Metric threshold should be between 0 and 1")
    
    # Validate scaling configuration
    if config.min_replicas > config.max_replicas:
        validation_results["valid"] = False
        validation_results["errors"].append("min_replicas cannot be greater than max_replicas")
    
    if config.max_replicas > 10:
        validation_results["warnings"].append("High replica count may consume significant resources")
    
    return validation_results


@router.get("/metrics/{deployment_id}")
async def get_deployment_metrics(
    deployment_id: str,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get runtime metrics for a deployment (CPU, memory, request count, etc.)
    """
    
    # Verify deployment access
    deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail="Deployment not found"
        )
    
    # Check project access
    project = db.query(Project).filter(
        Project.id == deployment.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )
    
    # Get metrics from Kubernetes/Prometheus
    try:
        import subprocess
        import json
        
        # Get pod metrics
        result = subprocess.run([
            "kubectl", "top", "pods", "-l", f"app=model-{deployment_id}",
            "-n", deployment.kubernetes_namespace, "--no-headers"
        ], capture_output=True, text=True)
        
        metrics = {
            "deployment_id": deployment_id,
            "status": deployment.status,
            "replicas": deployment.ready_replicas,
            "uptime_hours": (deployment.updated_at - deployment.created_at).total_seconds() / 3600,
            "resource_usage": {
                "cpu": "N/A",
                "memory": "N/A"
            },
            "request_metrics": {
                "total_requests": 0,
                "success_rate": 100.0,
                "avg_response_time_ms": 0
            }
        }
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines:
                # Parse first pod metrics
                parts = lines[0].split()
                if len(parts) >= 3:
                    metrics["resource_usage"]["cpu"] = parts[1]
                    metrics["resource_usage"]["memory"] = parts[2]
        
        return metrics
        
    except Exception as e:
        return {
            "deployment_id": deployment_id,
            "error": f"Failed to fetch metrics: {str(e)}"
        }
