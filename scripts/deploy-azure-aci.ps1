# Azure Container Instance Deployment Configuration
# This file defines the deployment of EasyML on Azure Container Instances

$resourceGroup = "easyml-rg"
$location = "East US"
$containerGroupName = "easyml-container-group"
$containerName = "easyml-api"
$imageName = "easyml:latest"
$dnsNameLabel = "easyml-api-$(Get-Random)"

# Environment variables for the container
$envVars = @(
    @{
        name = "POSTGRES_URL"
        value = "postgresql://postgres:postgres@localhost:5432/easyml"
    },
    @{
        name = "MONGO_URL" 
        value = "mongodb://localhost:27017/easyml"
    },
    @{
        name = "AZURE_STORAGE_CONNECTION_STRING"
        secureValue = $env:AZURE_STORAGE_CONNECTION_STRING
    },
    @{
        name = "AZURE_CONTAINER_NAME"
        value = "easyml-storage"
    },
    @{
        name = "ENVIRONMENT"
        value = "production"
    },
    @{
        name = "LOG_LEVEL"
        value = "INFO"
    },
    @{
        name = "SECRET_KEY"
        secureValue = $env:SECRET_KEY
    },
    @{
        name = "JWT_SECRET_KEY"
        secureValue = $env:JWT_SECRET_KEY
    }
)

# Create the container group
Write-Host "Creating Azure Container Instance..."

az container create `
    --resource-group $resourceGroup `
    --name $containerGroupName `
    --image $imageName `
    --cpu 2 `
    --memory 4 `
    --dns-name-label $dnsNameLabel `
    --ports 8000 `
    --environment-variables $envVars `
    --restart-policy OnFailure `
    --location $location

Write-Host "Container deployed successfully!"
Write-Host "API URL: http://$dnsNameLabel.$location.azurecontainer.io:8000"
