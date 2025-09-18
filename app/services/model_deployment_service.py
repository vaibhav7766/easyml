"""
Model Deployment Service with Docker isolation, GitHub automation, and model comparison
"""
import os
import yaml
import docker
import asyncio
import subprocess
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import uuid
import json

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_session
from app.models.sql_models import ModelVersion, Project, User, ModelDeployment, MLFlowRun
from app.schemas.deployment_schemas import (
    DeploymentConfig, ModelComparison, DeploymentRequest, 
    DeploymentResponse, DeploymentStatus, ModelSelectionCriteria
)
from app.core.enums import MetricType
from app.services.dvc_service import DVCService


class ModelDeploymentService:
    """Service for automated model deployment with comparison and containerization"""
    
    def __init__(self):
        self.settings = get_settings()
        self.dvc_service = DVCService()
        
        # Initialize Docker client with error handling
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            print(f"⚠️  Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        self.deployment_configs_path = Path("deployments")
        self.deployment_configs_path.mkdir(exist_ok=True)
    
    async def deploy_model(
        self,
        request: DeploymentRequest,
        db: Session
    ) -> DeploymentResponse:
        """
        Main deployment orchestrator with model comparison and selection
        """
        try:
            # 1. Get all models for the project
            models = await self._get_project_models(request.project_id, db)
            
            if not models:
                return DeploymentResponse(
                    deployment_id=str(uuid.uuid4()),
                    status="failed",
                    message="No models found for this project"
                )
            
            # 2. Compare models and select best one
            model_comparisons = await self._compare_models(
                models, 
                request.deployment_config.model_selection
            )
            
            # 3. Select model (manual override or automatic)
            selected_model = await self._select_model(
                model_comparisons,
                request.deployment_config.manual_model_id,
                request.deployment_config.model_selection
            )
            
            if not selected_model:
                return DeploymentResponse(
                    deployment_id=str(uuid.uuid4()),
                    status="failed",
                    message="No eligible model found for deployment",
                    model_comparisons=model_comparisons
                )
            
            # 4. Generate deployment configuration
            deployment_id = str(uuid.uuid4())
            deployment_yaml = await self._generate_deployment_yaml(
                deployment_id,
                selected_model,
                request.deployment_config
            )
            
            # 5. Build Docker container
            container_image = await self._build_model_container(
                deployment_id,
                selected_model,
                request.deployment_config
            )
            
            # 6. Deploy to Kubernetes (if not dry run)
            if not request.dry_run:
                k8s_namespace = await self._deploy_to_kubernetes(
                    deployment_yaml,
                    request.deployment_config.environment
                )
                
                # 7. Create deployment record
                await self._create_deployment_record(
                    deployment_id,
                    selected_model,
                    request.deployment_config,
                    container_image,
                    k8s_namespace,
                    db
                )
                
                # 8. Setup GitHub automation
                await self._setup_github_automation(
                    deployment_id,
                    request.project_id,
                    deployment_yaml
                )
            
            # 9. Generate API endpoint URL
            api_endpoint = self._generate_api_endpoint(
                deployment_id,
                request.deployment_config
            )
            
            return DeploymentResponse(
                deployment_id=deployment_id,
                status="deploying" if not request.dry_run else "dry_run_success",
                message="Model deployment initiated successfully" if not request.dry_run else "Dry run completed successfully",
                selected_model=selected_model,
                model_comparisons=model_comparisons,
                api_endpoint=api_endpoint,
                swagger_url=f"{api_endpoint}/docs",
                container_image=container_image,
                kubernetes_namespace=k8s_namespace if not request.dry_run else None,
                estimated_ready_time=datetime.utcnow() + timedelta(minutes=5)
            )
            
        except Exception as e:
            return DeploymentResponse(
                deployment_id=str(uuid.uuid4()),
                status="failed",
                message=f"Deployment failed: {str(e)}"
            )
    
    async def _get_project_models(
        self, 
        project_id: str, 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Get all models for a project with their metrics"""
        
        # Get models from PostgreSQL - convert project_id string to UUID
        try:
            project_uuid = uuid.UUID(project_id) if isinstance(project_id, str) else project_id
        except ValueError:
            print(f"⚠️ Invalid project_id format: {project_id}")
            return []
            
        models = db.query(ModelVersion).filter(
            ModelVersion.project_id == project_uuid,
            ModelVersion.status == "active"
        ).all()
        
        print(f"🔍 Found {len(models)} models for project {project_id}")
        
        enriched_models = []
        for model in models:
            print(f"🔍 Processing model: {model.name} (version: {model.version})")
            # Get MLflow metrics from PostgreSQL
            mlflow_run = db.query(MLFlowRun).filter(
                MLFlowRun.project_id == project_uuid
            ).first()
            
            model_data = {
                "id": str(model.id),
                "name": model.name,
                "version": model.version,
                "model_type": model.model_type,
                "storage_path": model.storage_path,
                "dvc_path": model.dvc_path,
                "size_bytes": model.size_bytes or 0,
                "performance_metrics": model.performance_metrics or {},
                "created_at": model.created_at,
                "mlflow_metrics": mlflow_run.metrics if mlflow_run else {}
            }
            
            # Merge metrics
            all_metrics = {**model_data["performance_metrics"], **model_data["mlflow_metrics"]}
            model_data["all_metrics"] = all_metrics
            
            enriched_models.append(model_data)
        
        return enriched_models
    
    async def _compare_models(
        self,
        models: List[Dict[str, Any]],
        criteria: ModelSelectionCriteria
    ) -> List[ModelComparison]:
        """Compare models based on selection criteria"""
        
        comparisons = []
        
        for model in models:
            # Calculate score based on primary metric
            primary_score = model["all_metrics"].get(criteria.primary_metric.value, 0)
            
            # Check if model meets minimum threshold
            meets_threshold = primary_score >= criteria.metric_threshold
            
            # Check secondary metrics
            meets_secondary = True
            for metric, threshold in criteria.secondary_metrics.items():
                if model["all_metrics"].get(metric.value, 0) < threshold:
                    meets_secondary = False
                    break
            
            # Check size constraints
            size_mb = model["size_bytes"] / (1024 * 1024)
            meets_size = True
            if criteria.max_model_size_mb and size_mb > criteria.max_model_size_mb:
                meets_size = False
            
            # Calculate overall score
            score = primary_score
            if criteria.prefer_latest:
                # Add bonus for newer models
                days_old = (datetime.utcnow() - model["created_at"]).days
                age_bonus = max(0, 1 - (days_old / 30))  # Bonus decreases over 30 days
                score += age_bonus * 0.1  # Small bonus for recency
            
            # Determine eligibility
            is_eligible = meets_threshold and meets_secondary and meets_size
            eligibility_reasons = []
            
            if not meets_threshold:
                eligibility_reasons.append(f"Primary metric {criteria.primary_metric.value} below threshold")
            if not meets_secondary:
                eligibility_reasons.append("Secondary metric thresholds not met")
            if not meets_size:
                eligibility_reasons.append("Model size exceeds limit")
            
            comparison = ModelComparison(
                model_id=model["id"],
                model_name=model["name"],
                version=model["version"],
                metrics=model["all_metrics"],
                score=score,
                created_at=model["created_at"],
                size_mb=size_mb,
                is_eligible=is_eligible,
                eligibility_reasons=eligibility_reasons
            )
            
            comparisons.append(comparison)
        
        # Sort by score (descending)
        comparisons.sort(key=lambda x: x.score, reverse=True)
        
        return comparisons
    
    async def _select_model(
        self,
        comparisons: List[ModelComparison],
        manual_model_id: Optional[str],
        criteria: ModelSelectionCriteria
    ) -> Optional[ModelComparison]:
        """Select the best model for deployment"""
        
        # Manual selection takes precedence
        if manual_model_id:
            for comparison in comparisons:
                if comparison.model_id == manual_model_id:
                    return comparison
        
        # Automatic selection - get first eligible model (highest score)
        for comparison in comparisons:
            if comparison.is_eligible:
                return comparison
        
        return None
    
    async def _generate_deployment_yaml(
        self,
        deployment_id: str,
        selected_model: ModelComparison,
        config: DeploymentConfig
    ) -> str:
        """Generate Kubernetes deployment YAML"""
        
        deployment_yaml = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"model-{deployment_id}",
                "namespace": f"easyml-{config.environment.value}",
                "labels": {
                    "app": f"model-{deployment_id}",
                    "project": config.project_id,
                    "model": selected_model.model_id,
                    "environment": config.environment.value
                }
            },
            "spec": {
                "replicas": config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": f"model-{deployment_id}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"model-{deployment_id}"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-api",
                            "image": f"easyml/model-{deployment_id}:latest",
                            "ports": [{
                                "containerPort": config.service_port
                            }],
                            "env": [
                                {"name": "MODEL_ID", "value": selected_model.model_id},
                                {"name": "API_PREFIX", "value": config.api_prefix},
                                {"name": "LOG_LEVEL", "value": config.log_level}
                            ] + [
                                {"name": k, "value": v} 
                                for k, v in config.container.env_vars.items()
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": config.container.cpu_request,
                                    "memory": config.container.memory_request
                                },
                                "limits": {
                                    "cpu": config.container.cpu_limit,
                                    "memory": config.container.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.container.health_check_path,
                                    "port": config.service_port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.container.readiness_probe_path,
                                    "port": config.service_port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Add service configuration
        service_yaml = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"model-{deployment_id}-service",
                "namespace": f"easyml-{config.environment.value}"
            },
            "spec": {
                "selector": {
                    "app": f"model-{deployment_id}"
                },
                "ports": [{
                    "port": 80,
                    "targetPort": config.service_port,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        # Add HPA configuration
        hpa_yaml = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"model-{deployment_id}-hpa",
                "namespace": f"easyml-{config.environment.value}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"model-{deployment_id}"
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": config.target_cpu_utilization
                        }
                    }
                }]
            }
        }
        
        # Combine all YAML configurations
        full_yaml = yaml.dump_all([deployment_yaml, service_yaml, hpa_yaml])
        
        # Save to file
        yaml_path = self.deployment_configs_path / f"{deployment_id}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(full_yaml)
        
        return str(yaml_path)
    
    async def _build_model_container(
        self,
        deployment_id: str,
        selected_model: ModelComparison,
        config: DeploymentConfig
    ) -> str:
        """Build Docker container for the model"""
        
        if not self.docker_available:
            # Return a mock image tag for development/testing
            image_tag = f"easyml/model-{deployment_id}:latest"
            print(f"⚠️  Docker not available - returning mock image: {image_tag}")
            return image_tag
        
        # Create temporary directory for Docker build
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download model from DVC
            model_path = await self._download_model(selected_model.model_id, temp_path)
            
            # Generate model API code
            api_code = self._generate_model_api_code(selected_model, config)
            
            # Write API code
            api_file = temp_path / "model_api.py"
            with open(api_file, 'w') as f:
                f.write(api_code)
            
            # Generate requirements.txt for container
            container_requirements = self._generate_container_requirements()
            req_file = temp_path / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write(container_requirements)
            
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(config)
            dockerfile = temp_path / "Dockerfile"
            with open(dockerfile, 'w') as f:
                f.write(dockerfile_content)
            
            # Build Docker image
            image_tag = f"easyml/model-{deployment_id}:latest"
            
            try:
                image = self.docker_client.images.build(
                    path=str(temp_path),
                    tag=image_tag,
                    rm=True
                )
                return image_tag
            except docker.errors.BuildError as e:
                raise Exception(f"Docker build failed: {e}")
    
    async def _download_model(self, model_id: str, temp_path: Path) -> Path:
        """Download model from DVC storage"""
        # This would integrate with your DVC service
        # For now, returning a placeholder path
        model_file = temp_path / "model.pkl"
        # Add actual DVC download logic here
        return model_file
    
    def _generate_model_api_code(
        self, 
        selected_model: ModelComparison, 
        config: DeploymentConfig
    ) -> str:
        """Generate FastAPI code for the model"""
        
        api_template = f'''
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = "./model.pkl"
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(
    title="EasyML Model API - {selected_model.model_name}",
    description="Auto-generated API for model {selected_model.model_name} v{selected_model.version}",
    version="{selected_model.version}",
    docs_url="/docs" if {config.enable_swagger} else None,
    redoc_url="/redoc" if {config.enable_swagger} else None
)

# Request/Response models
class PredictionRequest(BaseModel):
    data: Union[List[List[float]], List[Dict[str, Any]], Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    predictions: List[Union[float, int, str]]
    model_version: str = "{selected_model.version}"
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime

# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="{selected_model.version}",
        timestamp=datetime.utcnow()
    )

@app.get("/ready", response_model=HealthResponse) 
async def readiness_check():
    return HealthResponse(
        status="ready",
        model_loaded=model is not None,
        model_version="{selected_model.version}",
        timestamp=datetime.utcnow()
    )

# Model info endpoint
@app.get("/model/info")
async def model_info():
    return {{
        "model_name": "{selected_model.model_name}",
        "version": "{selected_model.version}",
        "model_id": "{selected_model.model_id}",
        "metrics": {selected_model.metrics},
        "size_mb": {selected_model.size_mb}
    }}

# Prediction endpoint
@app.post("{config.api_prefix}/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert input data to appropriate format
        if isinstance(request.data, dict):
            # Single prediction
            df = pd.DataFrame([request.data])
        elif isinstance(request.data[0], dict):
            # Multiple predictions with feature names
            df = pd.DataFrame(request.data)
        else:
            # Raw numeric data
            df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Convert predictions to list
        if hasattr(predictions, 'tolist'):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)
        
        return PredictionResponse(
            predictions=pred_list,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {{str(e)}}")

# Batch prediction endpoint
@app.post("{config.api_prefix}/predict/batch", response_model=PredictionResponse)
async def predict_batch(request: PredictionRequest):
    return await predict(request)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port={config.service_port},
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
'''
        return api_template
    
    def _generate_container_requirements(self) -> str:
        """Generate requirements.txt for container"""
        return '''
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
scikit-learn==1.4.0
joblib==1.3.2
pydantic==2.5.0
numpy==1.25.2
'''
    
    def _generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate Dockerfile for model container"""
        
        dockerfile_template = f'''
FROM {config.container.base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and API code
COPY model.pkl .
COPY model_api.py .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE {config.service_port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.service_port}/health || exit 1

# Run the application
CMD ["python", "model_api.py"]
'''
        return dockerfile_template
    
    async def _deploy_to_kubernetes(
        self, 
        yaml_path: str, 
        environment: str
    ) -> str:
        """Deploy to Kubernetes cluster"""
        
        namespace = f"easyml-{environment}"
        
        try:
            # Create namespace if it doesn't exist
            subprocess.run([
                "kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"
            ], check=False, capture_output=True)
            
            # Apply the deployment
            result = subprocess.run([
                "kubectl", "apply", "-f", yaml_path, "-n", namespace
            ], check=True, capture_output=True, text=True)
            
            return namespace
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Kubernetes deployment failed: {e.stderr}")
    
    async def _create_deployment_record(
        self,
        deployment_id: str,
        selected_model: ModelComparison,
        config: DeploymentConfig,
        container_image: str,
        k8s_namespace: str,
        db: Session
    ):
        """Create deployment record in database"""
        
        deployment_record = ModelDeployment(
            id=deployment_id,
            model_version_id=selected_model.model_id,
            environment=config.environment.value,
            endpoint_url=self._generate_api_endpoint(deployment_id, config),
            status="deploying",
            deployment_config={{
                "container_image": container_image,
                "kubernetes_namespace": k8s_namespace,
                "service_port": config.service_port,
                "replicas": config.min_replicas
            }}
        )
        
        db.add(deployment_record)
        db.commit()
    
    async def _setup_github_automation(
        self,
        deployment_id: str,
        project_id: str,
        deployment_yaml: str
    ):
        """Setup GitHub Actions for CI/CD automation"""
        
        # Create GitHub Actions workflow
        workflow_content = self._generate_github_workflow(deployment_id, project_id)
        
        # This would integrate with GitHub API to create/update workflows
        # For now, just save the workflow file
        workflow_path = self.deployment_configs_path / f"github-workflow-{deployment_id}.yml"
        with open(workflow_path, 'w') as f:
            f.write(workflow_content)
    
    def _generate_github_workflow(self, deployment_id: str, project_id: str) -> str:
        """Generate GitHub Actions workflow for automated deployment"""
        
        workflow_template = f'''
name: EasyML Model Deployment - {deployment_id}

on:
  push:
    paths:
      - 'models/{project_id}/**'
  workflow_dispatch:
    inputs:
      force_deploy:
        description: 'Force deployment'
        required: false
        default: 'false'

env:
  DEPLOYMENT_ID: {deployment_id}
  PROJECT_ID: {project_id}

jobs:
  model-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install DVC
        run: pip install dvc[azure]
      
      - name: Configure DVC
        run: |
          dvc remote add -d azure ${{{{ secrets.DVC_REMOTE_URL }}}}
          dvc remote modify azure connection_string ${{{{ secrets.AZURE_CONNECTION_STRING }}}}
      
      - name: Validate Model
        run: |
          python scripts/validate_model.py --deployment-id $DEPLOYMENT_ID
  
  build-and-deploy:
    needs: model-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker Image
        run: |
          docker build -t easyml/model-$DEPLOYMENT_ID:${{{{ github.sha }}}} .
          docker tag easyml/model-$DEPLOYMENT_ID:${{{{ github.sha }}}} easyml/model-$DEPLOYMENT_ID:latest
      
      - name: Push to Registry
        run: |
          echo ${{{{ secrets.DOCKER_PASSWORD }}}} | docker login -u ${{{{ secrets.DOCKER_USERNAME }}}} --password-stdin
          docker push easyml/model-$DEPLOYMENT_ID:latest
      
      - name: Deploy to Kubernetes
        run: |
          echo "${{{{ secrets.KUBECONFIG }}}}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
          kubectl apply -f deployments/$DEPLOYMENT_ID.yaml
      
      - name: Verify Deployment
        run: |
          kubectl wait --for=condition=available --timeout=300s deployment/model-$DEPLOYMENT_ID
'''
        return workflow_template
    
    def _generate_api_endpoint(
        self, 
        deployment_id: str, 
        config: DeploymentConfig
    ) -> str:
        """Generate API endpoint URL"""
        
        if config.domain:
            return f"https://{config.domain}{config.api_prefix}"
        else:
            # Use cluster internal URL
            namespace = f"easyml-{config.environment.value}"
            return f"http://model-{deployment_id}-service.{namespace}.svc.cluster.local{config.api_prefix}"
    
    async def get_deployment_status(
        self, 
        deployment_id: str,
        db: Session
    ) -> Optional[DeploymentStatus]:
        """Get current deployment status"""
        
        deployment = db.query(ModelDeployment).filter(
            ModelDeployment.id == deployment_id
        ).first()
        
        if not deployment:
            return None
        
        # Get Kubernetes status
        k8s_status = await self._get_kubernetes_status(deployment_id, deployment.environment)
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status=deployment.status,
            health_status=k8s_status.get("health", "unknown"),
            ready_replicas=k8s_status.get("ready_replicas", 0),
            total_replicas=k8s_status.get("total_replicas", 0),
            last_updated=deployment.updated_at
        )
    
    async def _get_kubernetes_status(
        self, 
        deployment_id: str, 
        environment: str
    ) -> Dict[str, Any]:
        """Get deployment status from Kubernetes"""
        
        try:
            namespace = f"easyml-{environment}"
            result = subprocess.run([
                "kubectl", "get", "deployment", f"model-{deployment_id}", 
                "-n", namespace, "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment_info = json.loads(result.stdout)
                status = deployment_info.get("status", {})
                
                return {
                    "health": "healthy" if status.get("readyReplicas", 0) > 0 else "unhealthy",
                    "ready_replicas": status.get("readyReplicas", 0),
                    "total_replicas": status.get("replicas", 0)
                }
            else:
                return {"health": "unknown", "ready_replicas": 0, "total_replicas": 0}
                
        except Exception:
            return {"health": "unknown", "ready_replicas": 0, "total_replicas": 0}
