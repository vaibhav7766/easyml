"""
Main API router for EasyML application
"""
from fastapi import APIRouter

from app.api.v1.endpoints import upload, visualization, preprocessing, training, auth, projects, dvc_endpoints

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(projects.router, tags=["Projects"])
api_router.include_router(dvc_endpoints.router, tags=["Data Version Control"])
api_router.include_router(upload.router, prefix="/upload", tags=["File Upload"])
api_router.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["Data Preprocessing"])
api_router.include_router(training.router, prefix="/training", tags=["Model Training"])
