"""
PostgreSQL Models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    projects = relationship("Project", back_populates="owner")
    experiments = relationship("MLExperiment", back_populates="user")
    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Project(Base):
    """Project model for organizing ML work"""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Project settings
    settings = Column(JSONB, default={})

    # Relationships
    owner = relationship("User", back_populates="projects")
    experiments = relationship("MLExperiment", back_populates="project")
    models = relationship("ModelVersion", back_populates="project")
    deployments = relationship("ModelDeployment", back_populates="project")


class MLExperiment(Base):
    """ML Experiment tracking (connected to MLflow)"""
    __tablename__ = "ml_experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # MLflow integration
    mlflow_experiment_id = Column(String(100), index=True)
    mlflow_run_id = Column(String(100), index=True)

    # Experiment details
    name = Column(String(200), nullable=False)
    description = Column(Text)
    model_type = Column(String(100), nullable=False)
    status = Column(String(50), default="running")  # running, completed, failed

    # Training configuration
    hyperparameters = Column(JSONB, default={})
    dataset_info = Column(JSONB, default={})

    # Results
    metrics = Column(JSONB, default={})
    artifacts_path = Column(String(500))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    project = relationship("Project", back_populates="experiments")
    user = relationship("User", back_populates="experiments")
    model_versions = relationship("ModelVersion", back_populates="experiment")


class ModelVersion(Base):
    """Model versioning with DVC integration"""
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("ml_experiments.id"), nullable=True)

    # MLflow integration
    mlflow_run_id = Column(String(100), index=True)

    # Model identification
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)

    # Storage information
    storage_path = Column(String(500), nullable=False)
    dvc_path = Column(String(500))
    mlflow_model_uri = Column(String(500))

    # Model metadata
    size_bytes = Column(Integer)
    checksum = Column(String(100))

    # Performance metrics
    performance_metrics = Column(JSONB, default={})

    # Status and lifecycle
    status = Column(String(50), default="active")
    is_production = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    project = relationship("Project", back_populates="models")
    experiment = relationship("MLExperiment", back_populates="model_versions")
    deployments = relationship("ModelDeployment", back_populates="model_version")


class DatasetVersion(Base):
    """Dataset versioning with DVC"""
    __tablename__ = "dataset_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)

    # Dataset identification
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)

    # Storage information
    storage_path = Column(String(500), nullable=False)
    dvc_path = Column(String(500))

    # Dataset metadata
    size_bytes = Column(Integer)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    checksum = Column(String(100))

    # Schema and statistics
    schema_info = Column(JSONB, default={})
    statistics = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    project = relationship("Project")


class ModelDeployment(Base):
    """Model deployment tracking"""
    __tablename__ = "model_deployments"

    id = Column(String(50), primary_key=True)  # Using deployment_id as primary key
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)

    # Deployment details
    environment = Column(String(50), nullable=False)
    endpoint_url = Column(String(500))
    status = Column(String(50), default="deploying")

    # Container and Kubernetes info
    container_image = Column(String(200))
    kubernetes_namespace = Column(String(100))

    # Configuration
    deployment_config = Column(JSONB, default={})

    # Metrics and monitoring
    metrics = Column(JSONB, default={})
    health_status = Column(String(50), default="unknown")
    ready_replicas = Column(Integer, default=0)
    total_replicas = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    model_version = relationship("ModelVersion", back_populates="deployments")
    project = relationship("Project", back_populates="deployments")


class RefreshToken(Base):
    """Refresh tokens for rotating refresh-token flow"""
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_hash = Column(String(128), index=True, nullable=False)  # sha256 hex
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    replaced_by = Column(UUID(as_uuid=True), nullable=True)  # id of new token if rotated
    user = relationship("User", back_populates="refresh_tokens")
