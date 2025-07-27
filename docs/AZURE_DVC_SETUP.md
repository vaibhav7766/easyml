# Azure Blob Storage Setup Guide for EasyML DVC

## Overview
This guide helps you configure Azure Blob Storage as the backend for DVC (Data Version Control) in your EasyML platform.

## Prerequisites
- Azure subscription
- Azure Storage Account
- Azure CLI (optional, for setup)

## Step 1: Create Azure Storage Account

### Option A: Using Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Storage accounts** → **Create**
3. Fill in the details:
   - **Subscription**: Your Azure subscription
   - **Resource group**: Create new or use existing
   - **Storage account name**: `easymlstorage` (must be globally unique)
   - **Region**: Choose closest to your application
   - **Performance**: Standard
   - **Redundancy**: LRS (Locally Redundant Storage) for development, GRS for production

### Option B: Using Azure CLI
```bash
# Login to Azure
az login

# Create resource group
az group create --name easyml-rg --location eastus

# Create storage account
az storage account create \
  --name easymlstorage \
  --resource-group easyml-rg \
  --location eastus \
  --sku Standard_LRS
```

## Step 2: Create Container for DVC

### Using Azure Portal
1. Go to your storage account
2. Navigate to **Containers** under **Data storage**
3. Click **+ Container**
4. Set container name: `easyml-data`
5. Set access level: **Private**

### Using Azure CLI
```bash
# Create container
az storage container create \
  --name easyml-data \
  --account-name easymlstorage
```

## Step 3: Get Connection String

### Using Azure Portal
1. Go to your storage account
2. Navigate to **Access keys** under **Security + networking**
3. Copy the **Connection string** from Key1 or Key2

### Using Azure CLI
```bash
# Get connection string
az storage account show-connection-string \
  --name easymlstorage \
  --resource-group easyml-rg \
  --output tsv
```

The connection string looks like:
```
DefaultEndpointsProtocol=https;AccountName=easymlstorage;AccountKey=XXXXX==;EndpointSuffix=core.windows.net
```

## Step 4: Update EasyML Configuration

Update your `.env` file with the Azure configuration:

```env
# DVC Configuration - Azure Blob Storage
DVC_REMOTE_URL="azure://easyml-data"
DVC_REMOTE_NAME="azure"
DVC_AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=easymlstorage;AccountKey=XXXXX==;EndpointSuffix=core.windows.net"
DVC_AZURE_CONTAINER_NAME="easyml-data"
```

**For Container Environments (Docker/Kubernetes):**
```env
# DVC Configuration - Azure Blob Storage (Container Environment)
DVC_REMOTE_URL="azure://easyml-data"
DVC_REMOTE_NAME="azure"
DVC_AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=easymlstorage;AccountKey=XXXXX==;EndpointSuffix=core.windows.net"
DVC_AZURE_CONTAINER_NAME="easyml-data"

# Container-specific settings
AZURE_STORAGE_ACCOUNT="easymlstorage"
AZURE_STORAGE_KEY="XXXXX=="
```

**Important**: Replace the values with your actual:
- Container name (`easyml-data`)
- Connection string (from Step 3)
- Storage account name and key for container authentication

## Step 5: Install Azure Dependencies

```bash
# Install DVC with Azure support
pip install dvc[azure]

# Or update requirements if not already done
pip install -r requirements.txt
```

## Step 6: Initialize and Test

```bash
# Check system status
python check_system_status.py

# Initialize database (this will also configure DVC)
python scripts/init_database.py

# Test DVC status
dvc remote list
```

## Step 7: Verify Configuration

Test that DVC can connect to Azure:

```bash
# Test Azure connection
dvc remote list

# Should show:
# azure    azure://easyml-data/easyml-data

# Test authentication
dvc push --dry-run
```

## Container Deployment with Azure Blob Storage

### Docker Environment Variables

When running in Docker containers, set these environment variables:

```bash
# Docker run example
docker run -d \
  -e POSTGRES_URL="postgresql://user:pass@postgres-host:5432/easyml_db" \
  -e MONGO_URL="mongodb://user:pass@mongo-host:27017" \
  -e MONGO_DB_NAME="easyml" \
  -e DVC_REMOTE_URL="azure://easyml-data" \
  -e DVC_AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=easymlstorage;AccountKey=XXXXX==;EndpointSuffix=core.windows.net" \
  -e DVC_AZURE_CONTAINER_NAME="easyml-data" \
  -e AZURE_STORAGE_ACCOUNT="easymlstorage" \
  -e AZURE_STORAGE_KEY="XXXXX==" \
  -p 8000:8000 \
  easyml-api
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  easyml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/easyml_db
      - MONGO_URL=mongodb://user:pass@mongo:27017
      - MONGO_DB_NAME=easyml
      - DVC_REMOTE_URL=azure://easyml-data
      - DVC_AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=easymlstorage;AccountKey=XXXXX==;EndpointSuffix=core.windows.net
      - DVC_AZURE_CONTAINER_NAME=easyml-data
      - AZURE_STORAGE_ACCOUNT=easymlstorage
      - AZURE_STORAGE_KEY=XXXXX==
      - SECRET_KEY=your-secret-key
    volumes:
      - ./dvc_storage:/app/dvc_storage
      - ./uploads:/app/uploads
    depends_on:
      - postgres
      - mongo

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=easyml_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mongo:
    image: mongo:7
    environment:
      - MONGO_INITDB_ROOT_USERNAME=user
      - MONGO_INITDB_ROOT_PASSWORD=pass
    volumes:
      - mongo_data:/data/db

volumes:
  postgres_data:
  mongo_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: easyml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: easyml-api
  template:
    metadata:
      labels:
        app: easyml-api
    spec:
      containers:
      - name: easyml-api
        image: easyml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DVC_REMOTE_URL
          value: "azure://easyml-data"
        - name: DVC_AZURE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: connection-string
        - name: DVC_AZURE_CONTAINER_NAME
          value: "easyml-data"
        - name: AZURE_STORAGE_ACCOUNT
          value: "easymlstorage"
        - name: AZURE_STORAGE_KEY
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: storage-key
        volumeMounts:
        - name: dvc-storage
          mountPath: /app/dvc_storage
        - name: uploads
          mountPath: /app/uploads
      volumes:
      - name: dvc-storage
        emptyDir: {}
      - name: uploads
        emptyDir: {}
---
apiVersion: v1
kind: Secret
metadata:
  name: azure-storage-secret
type: Opaque
data:
  connection-string: <base64-encoded-connection-string>
  storage-key: <base64-encoded-storage-key>
```

## Security Best Practices

### 1. Use Managed Identity (Recommended for Production)

Instead of connection strings, use Azure Managed Identity:

```bash
# Configure managed identity
az storage account update \
  --name easymlstorage \
  --resource-group easyml-rg \
  --assign-identity

# Grant permissions to your application
az role assignment create \
  --role "Storage Blob Data Contributor" \
  --assignee YOUR_APP_PRINCIPAL_ID \
  --scope /subscriptions/YOUR_SUBSCRIPTION/resourceGroups/easyml-rg/providers/Microsoft.Storage/storageAccounts/easymlstorage
```

### 2. Use SAS Tokens (Alternative)

Create a SAS token with limited permissions:

```bash
# Generate SAS token (valid for 1 year)
az storage container generate-sas \
  --name easyml-data \
  --account-name easymlstorage \
  --permissions dlrw \
  --expiry 2026-12-31T23:59:59Z \
  --output tsv
```

### 3. Network Security

Restrict access to your storage account:

```bash
# Allow access only from specific IP ranges
az storage account network-rule add \
  --account-name easymlstorage \
  --resource-group easyml-rg \
  --ip-address YOUR_SERVER_IP
```

## Monitoring and Costs

### Monitor Storage Usage
- Go to **Azure Portal** → **Storage account** → **Metrics**
- Monitor **Used capacity** and **Transaction count**

### Cost Optimization
- Use **Cool** or **Archive** tiers for older model versions
- Set up lifecycle policies to automatically move old data
- Monitor transaction costs (DVC operations)

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```bash
   # Check connection string format
   echo $DVC_AZURE_CONNECTION_STRING
   
   # Test with Azure CLI
   az storage container list --connection-string "$DVC_AZURE_CONNECTION_STRING"
   ```

2. **Container Not Found**
   ```bash
   # List containers
   az storage container list --account-name easymlstorage
   ```

3. **Permission Denied**
   ```bash
   # Check account permissions
   az storage account show --name easymlstorage --query primaryEndpoints
   ```

### DVC Azure Commands

```bash
# List DVC remotes
dvc remote list

# Test connection
dvc status

# Push data to Azure
dvc push

# Pull data from Azure
dvc pull

# Check DVC configuration
dvc config list
```

## Production Considerations

1. **Backup Strategy**: Enable geo-redundant storage
2. **Access Control**: Use RBAC and managed identities
3. **Monitoring**: Set up Azure Monitor alerts
4. **Compliance**: Configure data residency requirements
5. **Performance**: Use premium storage for high-throughput scenarios

## Example Azure Configuration

```env
# Production Azure Configuration
DVC_REMOTE_URL="azure://easyml-prod-data/models"
DVC_REMOTE_NAME="azure-prod"
DVC_AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=easymlprodstorage;AccountKey=PRODUCTION_KEY==;EndpointSuffix=core.windows.net"
DVC_AZURE_CONTAINER_NAME="easyml-prod-data"
```

Your EasyML platform will now automatically use Azure Blob Storage for all DVC operations with complete user/project isolation! 🚀
