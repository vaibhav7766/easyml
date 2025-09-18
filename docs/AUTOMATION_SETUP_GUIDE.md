# EasyML Complete CI/CD Automation Setup Guide

## Overview

This guide will help you set up complete automation for your EasyML project using:
- **DVC** for ML pipeline management
- **GitHub Actions** for CI/CD automation  
- **Azure Container Instances** for deployment
- **Docker** for containerization

## Prerequisites

1. **GitHub Repository**: Your EasyML code in a GitHub repository
2. **Azure Account**: With active subscription
3. **Azure CLI**: Installed and configured locally
4. **Docker**: Installed locally for testing

## Step 1: Azure Resource Setup

### 1.1 Create Resource Group
```bash
az group create --name easyml-rg --location eastus
```

### 1.2 Create Azure Container Registry (ACR)
```bash
az acr create --resource-group easyml-rg --name youracrname --sku Basic --admin-enabled true
```

### 1.3 Create Storage Account for DVC
```bash
az storage account create \
  --name yourstorageaccount \
  --resource-group easyml-rg \
  --location eastus \
  --sku Standard_LRS

az storage container create \
  --name dvc-storage \
  --account-name yourstorageaccount
```

### 1.4 Create Service Principal for GitHub Actions
```bash
az ad sp create-for-rbac --name "easyml-github-actions" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/easyml-rg \
  --sdk-auth
```

Save the JSON output - you'll need it for GitHub Secrets.

## Step 2: GitHub Repository Configuration

### 2.1 Repository Secrets

Go to your GitHub repository → Settings → Secrets and Variables → Actions

Add these secrets:

#### Azure Credentials
- `AZURE_CREDENTIALS`: The entire JSON from service principal creation
- `AZURE_CLIENT_ID`: From the JSON output
- `AZURE_CLIENT_SECRET`: From the JSON output  
- `AZURE_TENANT_ID`: From the JSON output
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

#### Container Registry
- `ACR_LOGIN_SERVER`: youracrname.azurecr.io
- `ACR_USERNAME`: Get from `az acr credential show --name youracrname`
- `ACR_PASSWORD`: Get from `az acr credential show --name youracrname`

#### Storage for DVC
- `AZURE_STORAGE_ACCOUNT`: yourstorageaccount
- `AZURE_STORAGE_KEY`: Get from `az storage account keys list`
- `AZURE_STORAGE_CONTAINER`: dvc-storage

#### Database Configuration
- `POSTGRES_HOST`: Your PostgreSQL host
- `POSTGRES_DB`: easyml
- `POSTGRES_USER`: easyml_user
- `POSTGRES_PASSWORD`: Your secure password
- `MONGODB_URI`: Your MongoDB connection string

#### API Configuration
- `SECRET_KEY`: Generate with `openssl rand -hex 32`
- `API_HOST`: 0.0.0.0
- `API_PORT`: 8000

### 2.2 Environment Variables

Copy `.env.template` to `.env` and fill in your values:
```bash
cp .env.template .env
# Edit .env with your actual values
```

## Step 3: DVC Setup

### 3.1 Initialize DVC (if not already done)
```bash
dvc init --no-scm
git add .dvc/
git commit -m "Initialize DVC"
```

### 3.2 Configure Azure Storage Remote
```bash
dvc remote add -d azure_storage azure://dvc-storage
dvc remote modify azure_storage account_name yourstorageaccount

# Set connection string
dvc remote modify azure_storage connection_string "DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=your_key;EndpointSuffix=core.windows.net"
```

### 3.3 Add Sample Data (if you have training data)
```bash
# Add your training data
dvc add datasets/train.csv
git add datasets/train.csv.dvc datasets/.gitignore
git commit -m "Add training data"

# Push to remote storage
dvc push
```

## Step 4: Pipeline Configuration

### 4.1 Parameters Setup

The `params.yaml` file is already configured. Adjust parameters as needed:

```yaml
# Edit params.yaml
data_preparation:
  test_size: 0.2
  random_state: 42

model_training:
  model_type: "random_forest"
  hyperparameters:
    random_forest:
      n_estimators: 100
      max_depth: 10
```

### 4.2 Test Pipeline Locally

```bash
# Test data preparation
python scripts/data_preparation.py --input datasets/your_data.csv --output data/prepared

# Test feature engineering  
python scripts/feature_engineering.py --input data/prepared/prepared_data.csv --output features

# Test model training
python scripts/model_training.py --input features/engineered_data.csv --output models
```

## Step 5: Docker Testing

### 5.1 Build and Test Locally
```bash
# Build the image
docker-compose build

# Test locally
docker-compose up -d

# Check if API is running
curl http://localhost:8000/health

# Stop services
docker-compose down
```

## Step 6: Deployment Configuration

### 6.1 Update deployment configs

Edit `deployment-configs/production-deployment.yaml`:
```yaml
name: easyml-api-prod
resource_group: easyml-rg
location: eastus
dns_name_label: easyml-api-prod-unique  # Must be globally unique
ports:
  - 8000
environment_variables:
  ENV: production
  API_HOST: "0.0.0.0"
```

## Step 7: GitHub Actions Workflow

The workflow file `.github/workflows/ci-cd-pipeline.yml` is already configured. It will:

1. **Test Stage**: Run linting and tests
2. **Build Stage**: Build and push Docker images
3. **DVC Stage**: Execute ML pipeline
4. **Deploy Stage**: Deploy to Azure Container Instances
5. **Notify Stage**: Send deployment notifications

## Step 8: Triggering the Pipeline

### 8.1 Automatic Triggers

The pipeline triggers on:
- Push to `main` branch
- Pull requests to `main` branch
- Manual dispatch (workflow_dispatch)

### 8.2 Manual Trigger

Go to GitHub → Actions → CI/CD Pipeline → Run workflow

## Step 9: Monitoring and Validation

### 9.1 Check Deployment Status

```bash
# Check container status
az container show --resource-group easyml-rg --name easyml-api

# Get public IP
az container show --resource-group easyml-rg --name easyml-api --query ipAddress.ip
```

### 9.2 Test Deployed API

```bash
# Health check
curl http://your-container-ip:8000/health

# API documentation
curl http://your-container-ip:8000/docs
```

### 9.3 Monitor Logs

```bash
# View container logs
az container logs --resource-group easyml-rg --name easyml-api

# Stream logs
az container logs --resource-group easyml-rg --name easyml-api --follow
```

## Step 10: Advanced Configuration

### 10.1 Custom Domain (Optional)

```bash
# Create DNS zone
az network dns zone create --resource-group easyml-rg --name yourdomain.com

# Add A record pointing to container IP
az network dns record-set a add-record \
  --resource-group easyml-rg \
  --zone-name yourdomain.com \
  --record-set-name api \
  --ipv4-address your-container-ip
```

### 10.2 SSL/HTTPS (Optional)

Update `docker-compose.yml` to include SSL certificates and configure nginx for HTTPS.

### 10.3 Scaling (Optional)

Consider upgrading to Azure Container Apps or AKS for production scaling needs.

## Troubleshooting

### Common Issues

1. **GitHub Actions Failing**:
   - Check secrets are correctly set
   - Verify Azure credentials have proper permissions
   - Check workflow logs for specific errors

2. **Docker Build Failing**:
   - Ensure all dependencies are in requirements.txt
   - Check Dockerfile syntax
   - Verify base image compatibility

3. **Azure Deployment Failing**:
   - Check resource group and ACR exist
   - Verify container image was pushed successfully
   - Check Azure resource quotas

4. **DVC Pipeline Failing**:
   - Ensure data files exist and are accessible
   - Check DVC remote configuration
   - Verify pipeline dependencies are correct

### Logs and Debugging

```bash
# GitHub Actions logs: Check in GitHub Actions tab
# Azure Container logs: az container logs --resource-group easyml-rg --name easyml-api
# Local Docker logs: docker-compose logs
# DVC pipeline logs: Check mlruns/ directory
```

## Next Steps

1. **Monitoring**: Set up Azure Monitor for production monitoring
2. **Alerts**: Configure alerts for API health and performance
3. **Backup**: Implement backup strategies for models and data
4. **Security**: Review security best practices and implement additional measures
5. **Performance**: Monitor and optimize API performance

## Support

For issues or questions:
1. Check GitHub Actions logs
2. Review Azure resource status
3. Check container logs
4. Verify environment configurations

Your EasyML project is now fully automated with CI/CD pipeline, containerized deployment, and ML pipeline automation!
