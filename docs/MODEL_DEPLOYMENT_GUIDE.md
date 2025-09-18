# EasyML Model Deployment System

## Overview

The EasyML Model Deployment System provides automated model deployment with Docker isolation, GitHub automation, and intelligent model comparison. This system allows you to deploy trained models as REST APIs with minimal configuration.

## 🚀 Key Features

- **Automated Model Comparison**: Compare models by evaluation metrics and select the best one
- **Docker Isolation**: Each model runs in its own isolated Docker container
- **Kubernetes Orchestration**: Auto-scaling, health checks, and high availability
- **GitHub Automation**: CI/CD pipelines for automated deployments
- **YAML Configuration**: Declarative configuration for different environments
- **Multi-Environment Support**: Development, staging, and production environments
- **Real-time Monitoring**: Health checks, metrics, and deployment status

## 📋 Prerequisites

1. **Docker**: Installed and running
2. **Kubernetes**: Cluster access with kubectl configured
3. **GitHub**: Repository with Actions enabled
4. **Azure Blob Storage**: For DVC model storage
5. **Database**: PostgreSQL and MongoDB running

## 🛠️ Setup Instructions

### 1. Database Migration

Run the migration script to add deployment tables:

```bash
python scripts/migrate_deployment_tables.py
```

### 2. Configure Environment Variables

Add these environment variables to your `.env` file:

```env
# Docker Registry
DOCKER_REGISTRY=your-docker-registry.com
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password

# Kubernetes
KUBECONFIG_BASE64=your-base64-encoded-kubeconfig

# DVC (already configured)
DVC_REMOTE_URL=your-azure-blob-url
AZURE_CONNECTION_STRING=your-azure-connection-string
```

### 3. GitHub Secrets

Configure these secrets in your GitHub repository:

- `DOCKER_USERNAME` - Docker registry username
- `DOCKER_PASSWORD` - Docker registry password
- `KUBECONFIG` - Base64 encoded Kubernetes config
- `DVC_REMOTE_URL` - DVC remote storage URL
- `AZURE_CONNECTION_STRING` - Azure storage connection string
- `DATABASE_URL` - PostgreSQL connection string
- `MONGODB_URL` - MongoDB connection string

## 🎯 Usage Guide

### 1. Deploy via API

#### Basic Deployment

```python
import requests

# Deployment configuration
config = {
    "project_id": "your-project-id",
    "deployment_config": {
        "model_selection": {
            "primary_metric": "accuracy",
            "metric_threshold": 0.8,
            "secondary_metrics": {
                "precision": 0.7,
                "recall": 0.7
            },
            "prefer_latest": True,
            "max_model_size_mb": 100
        },
        "environment": "staging",
        "container": {
            "base_image": "python:3.10-slim",
            "cpu_request": "100m",
            "cpu_limit": "500m",
            "memory_request": "128Mi",
            "memory_limit": "512Mi"
        },
        "service_port": 8000,
        "api_prefix": "/api/v1",
        "min_replicas": 1,
        "max_replicas": 3,
        "enable_swagger": True
    },
    "dry_run": False
}

# Deploy the model
response = requests.post(
    "http://localhost:8000/v1/deployments/deploy",
    json=config,
    headers={"Authorization": "Bearer your-jwt-token"}
)

deployment = response.json()
print(f"Deployment ID: {deployment['deployment_id']}")
print(f"API Endpoint: {deployment['api_endpoint']}")
```

#### Manual Model Selection

```python
# First, compare models to see available options
response = requests.post(
    f"http://localhost:8000/v1/deployments/compare-models/{project_id}",
    json={
        "primary_metric": "f1_score",
        "metric_threshold": 0.85,
        "secondary_metrics": {
            "accuracy": 0.8,
            "precision": 0.8
        }
    },
    headers={"Authorization": "Bearer your-jwt-token"}
)

models = response.json()
print("Available models:")
for model in models:
    print(f"- {model['model_name']} v{model['version']}: {model['score']:.3f}")

# Deploy specific model
config["deployment_config"]["manual_model_id"] = "specific-model-id"
```

### 2. Deploy via YAML Configuration

Create a deployment configuration file:

```yaml
# deployment-config.yaml
deployment:
  project_id: "your-project-id"
  environment: "production"
  
  model_selection:
    primary_metric: "accuracy"
    metric_threshold: 0.9
    secondary_metrics:
      precision: 0.85
      recall: 0.85
    prefer_latest: true
    max_model_size_mb: 500
  
  container:
    base_image: "python:3.10-slim"
    cpu_request: "200m"
    cpu_limit: "1000m"
    memory_request: "256Mi"
    memory_limit: "1Gi"
  
  scaling:
    min_replicas: 2
    max_replicas: 10
    target_cpu_utilization: 70
```

### 3. Monitor Deployments

#### Check Deployment Status

```python
response = requests.get(
    f"http://localhost:8000/v1/deployments/status/{deployment_id}",
    headers={"Authorization": "Bearer your-jwt-token"}
)

status = response.json()
print(f"Status: {status['status']}")
print(f"Health: {status['health_status']}")
print(f"Replicas: {status['ready_replicas']}/{status['total_replicas']}")
```

#### List All Deployments

```python
response = requests.get(
    f"http://localhost:8000/v1/deployments/list/{project_id}",
    headers={"Authorization": "Bearer your-jwt-token"}
)

deployments = response.json()
for deployment in deployments:
    print(f"{deployment['deployment_id']}: {deployment['status']}")
```

#### Get Runtime Metrics

```python
response = requests.get(
    f"http://localhost:8000/v1/deployments/metrics/{deployment_id}",
    headers={"Authorization": "Bearer your-jwt-token"}
)

metrics = response.json()
print(f"CPU Usage: {metrics['resource_usage']['cpu']}")
print(f"Memory Usage: {metrics['resource_usage']['memory']}")
```

### 4. Use Deployed Model

Once deployed, your model is available as a REST API:

```python
# Make predictions
prediction_data = {
    "data": [
        {"feature1": 1.0, "feature2": 2.5, "feature3": 0.8},
        {"feature1": 1.5, "feature2": 3.0, "feature3": 1.2}
    ]
}

response = requests.post(
    f"{api_endpoint}/predict",
    json=prediction_data
)

predictions = response.json()
print(f"Predictions: {predictions['predictions']}")
```

## 🔧 Configuration Options

### Model Selection Criteria

| Parameter | Description | Example |
|-----------|-------------|---------|
| `primary_metric` | Main metric for comparison | `"accuracy"`, `"f1_score"` |
| `metric_threshold` | Minimum acceptable value | `0.8` |
| `secondary_metrics` | Additional metric constraints | `{"precision": 0.7}` |
| `prefer_latest` | Prefer newer models | `true` |
| `max_model_size_mb` | Size limit in MB | `100` |

### Container Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `base_image` | Docker base image | `"python:3.10-slim"` |
| `cpu_request` | CPU request | `"100m"` |
| `cpu_limit` | CPU limit | `"500m"` |
| `memory_request` | Memory request | `"128Mi"` |
| `memory_limit` | Memory limit | `"512Mi"` |
| `env_vars` | Environment variables | `{"LOG_LEVEL": "INFO"}` |

### Scaling Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `min_replicas` | Minimum pod count | `1` |
| `max_replicas` | Maximum pod count | `5` |
| `target_cpu_utilization` | CPU threshold for scaling | `70` |
| `auto_scaling` | Enable auto-scaling | `true` |

## 🚨 Troubleshooting

### Common Issues

1. **Deployment Stuck in "deploying" Status**
   ```bash
   kubectl get pods -n easyml-staging
   kubectl describe pod <pod-name> -n easyml-staging
   ```

2. **Model API Not Responding**
   ```bash
   kubectl logs deployment/model-<deployment-id> -n easyml-staging
   ```

3. **Docker Build Failures**
   - Check model file availability
   - Verify base image compatibility
   - Check resource limits

### Health Check Endpoints

All deployed models provide these endpoints:

- `GET /health` - Basic health check
- `GET /ready` - Readiness probe
- `GET /model/info` - Model metadata
- `GET /docs` - Swagger documentation (if enabled)

### Logs and Monitoring

```bash
# View deployment logs
kubectl logs -f deployment/model-<deployment-id> -n easyml-staging

# Check resource usage
kubectl top pods -n easyml-staging

# View deployment events
kubectl describe deployment model-<deployment-id> -n easyml-staging
```

## 🔒 Security Considerations

1. **API Authentication**: All endpoints require JWT authentication
2. **Container Security**: Runs as non-root user
3. **Network Policies**: Kubernetes network isolation
4. **Secrets Management**: Use Kubernetes secrets for sensitive data
5. **Image Scanning**: Scan container images for vulnerabilities

## 📈 Performance Optimization

1. **Resource Limits**: Set appropriate CPU/memory limits
2. **Auto-scaling**: Configure HPA for traffic spikes
3. **Caching**: Implement model prediction caching
4. **Load Balancing**: Use Kubernetes service load balancing
5. **Monitoring**: Set up Prometheus/Grafana monitoring

## 🔄 CI/CD Integration

### Automatic Deployment on Model Training

1. **Trigger**: When new model is trained and meets criteria
2. **GitHub Actions**: Automatically builds and deploys
3. **Testing**: Validates deployment before going live
4. **Rollback**: Automatic rollback on failure

### Manual Deployment

```bash
# Trigger deployment via GitHub Actions
gh workflow run model-deployment.yml \
  -f project_id="your-project-id" \
  -f environment="production" \
  -f manual_model_id="model-123"
```

## 📞 Support

For issues and questions:

1. Check the logs using kubectl commands above
2. Review the API documentation at `/docs`
3. Check deployment status via API endpoints
4. Review GitHub Actions logs for CI/CD issues

## 🎯 Example Workflows

### Development Workflow

1. Train model locally
2. Deploy to development environment
3. Test API endpoints
4. Review metrics and logs
5. Promote to staging

### Production Workflow

1. Model passes staging tests
2. GitHub Actions triggers production deployment
3. Blue-green deployment strategy
4. Health checks validate deployment
5. Traffic gradually shifted to new version
6. Monitor metrics and rollback if needed

This deployment system provides a complete solution for automated model deployment with enterprise-grade features like monitoring, scaling, and CI/CD integration.
