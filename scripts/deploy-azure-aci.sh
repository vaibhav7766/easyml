#!/bin/bash

# Azure Container Instance Deployment Script for EasyML
# This script deploys the EasyML application to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="easyml-rg"
LOCATION="eastus"
CONTAINER_GROUP_NAME="easyml-container-group"
CONTAINER_NAME="easyml-api"
IMAGE_NAME="easyml:latest"
DNS_NAME_LABEL="easyml-api-$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying EasyML to Azure Container Instances${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Login to Azure (if not already logged in)
echo -e "${YELLOW}📋 Checking Azure login status...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}🔐 Please login to Azure...${NC}"
    az login
fi

# Create resource group if it doesn't exist
echo -e "${YELLOW}📦 Creating resource group if it doesn't exist...${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Build and push Docker image to Azure Container Registry (optional)
if [ "$1" = "with-acr" ]; then
    ACR_NAME="easymlacr$(date +%s)"
    echo -e "${YELLOW}🏗️ Creating Azure Container Registry...${NC}"
    
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --location $LOCATION

    echo -e "${YELLOW}🔨 Building and pushing Docker image...${NC}"
    az acr build \
        --registry $ACR_NAME \
        --image $IMAGE_NAME \
        --file scripts/Dockerfile .

    IMAGE_NAME="$ACR_NAME.azurecr.io/$IMAGE_NAME"
fi

# Deploy container to Azure Container Instances
echo -e "${YELLOW}🚀 Deploying container to Azure...${NC}"

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_GROUP_NAME \
    --image $IMAGE_NAME \
    --cpu 2 \
    --memory 4 \
    --dns-name-label $DNS_NAME_LABEL \
    --ports 8000 \
    --environment-variables \
        POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/easyml" \
        MONGO_URL="mongodb://localhost:27017/easyml" \
        AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
        AZURE_CONTAINER_NAME="easyml-storage" \
        ENVIRONMENT="production" \
        LOG_LEVEL="INFO" \
        SECRET_KEY="$SECRET_KEY" \
        JWT_SECRET_KEY="$JWT_SECRET_KEY" \
    --restart-policy OnFailure \
    --location $LOCATION \
    --output table

# Get the public IP and URL
echo -e "${GREEN}✅ Deployment completed!${NC}"
echo -e "${GREEN}🌐 Getting deployment information...${NC}"

FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_GROUP_NAME \
    --query ipAddress.fqdn \
    --output tsv)

IP=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_GROUP_NAME \
    --query ipAddress.ip \
    --output tsv)

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 EasyML API Deployed Successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📍 API URL:${NC} ${GREEN}http://${FQDN}:8000${NC}"
echo -e "${YELLOW}🔗 API Documentation:${NC} ${GREEN}http://${FQDN}:8000/docs${NC}"
echo -e "${YELLOW}🏥 Health Check:${NC} ${GREEN}http://${FQDN}:8000/health${NC}"
echo -e "${YELLOW}📊 Public IP:${NC} ${GREEN}${IP}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Test the deployment
echo -e "${YELLOW}🧪 Testing the deployment...${NC}"
sleep 30  # Wait for container to start

if curl -f "http://${FQDN}:8000/health" &> /dev/null; then
    echo -e "${GREEN}✅ Health check passed! API is running.${NC}"
else
    echo -e "${RED}❌ Health check failed. Please check container logs.${NC}"
    echo -e "${YELLOW}🔍 View logs with: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_GROUP_NAME${NC}"
fi

echo -e "${GREEN}🎯 Deployment script completed!${NC}"
