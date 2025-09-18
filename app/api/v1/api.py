"""
Main API router for EasyML application
"""
from fastapi import APIRouter

from app.api.v1.endpoints import upload, visualization, preprocessing, training, auth, projects, dvc_endpoints, deployment

api_router = APIRouter()

# API root endpoint
@api_router.get("/")
async def api_root():
    """API v1 root endpoint with navigation"""
    return {
        "message": "EasyML API v1",
        "documentation": "/docs",
        "endpoints": {
            "authentication": "/api/v1/auth",
            "projects": "/api/v1/projects", 
            "data_version_control": "/api/v1/dvc",
            "file_upload": "/api/v1/upload",
            "visualization": "/api/v1/visualization",
            "preprocessing": "/api/v1/preprocessing",
            "training": "/api/v1/training",
            "deployment": "/api/v1/deployments"
        }
    }

# Include all endpoint routers
api_router.include_router(auth.router)
api_router.include_router(projects.router)
api_router.include_router(dvc_endpoints.router)
api_router.include_router(upload.router, prefix="/upload", tags=["File Upload"])
api_router.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["Data Preprocessing"])
api_router.include_router(training.router, prefix="/training", tags=["Model Training"])
api_router.include_router(deployment.router)
