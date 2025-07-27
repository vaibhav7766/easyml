"""
Enhanced Model Training Service with Database Integration and DVC
"""
import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from pymongo.database import Database

from app.services.model_training import ModelTrainingService as BaseModelTrainingService
from app.services.dvc_service import DVCService
from app.models.sql_models import MLExperiment, ModelVersion, User, Project
from app.models.mongo_schemas import ModelSessionDocument, DVCMetadataDocument, MLFlowRunDocument
import mlflow
import joblib


class EnhancedModelTrainingService(BaseModelTrainingService):
    """Enhanced model training service with database integration and DVC"""
    
    def __init__(
        self, 
        session_id: str,
        user: Optional[User] = None,
        project: Optional[Project] = None,
        db_session: Optional[Session] = None,
        mongo_db: Optional[Database] = None
    ):
        super().__init__()
        self.session_id = session_id
        self.user = user
        self.project = project
        self.db_session = db_session
        self.mongo_db = mongo_db
        self.ml_experiment = None
        self.model_version = None
        
        # Initialize DVC service
        self.dvc_service = DVCService()
    
    async def train_model_with_persistence(
        self,
        data: Any,
        target_column: str,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        use_cross_validation: bool = True,
        cv_folds: int = 5,
        experiment_name: str = "easyml_experiment",
        auto_version: bool = True
    ) -> Dict[str, Any]:
        """Train model with database persistence"""
        
        # Start MLflow run
        mlflow_experiment_id = None
        mlflow_run_id = None
        
        if self.project:
            experiment_name = f"project_{self.project.id}_{experiment_name}"
        
        try:
            # Set MLflow experiment
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                mlflow_run_id = run.info.run_id
                mlflow_experiment_id = run.info.experiment_id
                
                # Train the model using parent class
                results = super().train_model(
                    data=data,
                    target_column=target_column,
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    test_size=test_size,
                    use_cross_validation=use_cross_validation,
                    cv_folds=cv_folds,
                    experiment_name=experiment_name
                )
                
                # Save model to project folder if project exists
                if self.project:
                    model_path = await self._save_model_to_project(results, mlflow_run_id)
                    
                    # Version with DVC if auto_version is enabled
                    if auto_version and model_path:
                        await self._version_model_with_dvc(model_path, results, mlflow_run_id)
                
                # Store session in MongoDB
                await self._store_session_data(results, mlflow_run_id, mlflow_experiment_id)
                
                # Store experiment in PostgreSQL
                await self._store_experiment_data(results, mlflow_run_id, mlflow_experiment_id)
                
                # Store MLflow run metadata in MongoDB
                await self._store_mlflow_run_data(run, results)
                
                return results
                
        except Exception as e:
            # Log error and re-raise
            if self.mongo_db:
                await self._log_error(str(e), mlflow_run_id)
            raise e
    
    async def _save_model_to_project(self, results: Dict[str, Any], mlflow_run_id: str) -> Optional[str]:
        """Save model to project folder structure"""
        if not self.project or not self.model:
            return None
        
        # Create model directory structure with user/project isolation
        user_id = str(self.user.id) if self.user else "unknown"
        project_id = str(self.project.id)
        
        model_dir = self.dvc_service.get_user_project_path(user_id, project_id, "models")
        model_subdir = model_dir / f"model_{uuid.uuid4()}"
        model_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_subdir / "model.joblib"
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'is_classifier': self.is_classifier,
            'hyperparameters': results.get('hyperparameters', {}),
            'metrics': results.get('test_metrics', {}),
            'mlflow_run_id': mlflow_run_id,
            'user_id': user_id,
            'project_id': project_id,
            'session_id': self.session_id
        }, model_path)
        
        return str(model_path)
    
    async def _version_model_with_dvc(
        self, 
        model_path: str, 
        results: Dict[str, Any], 
        mlflow_run_id: str
    ):
        """Version the model using DVC"""
        if not self.user or not self.project:
            return
        
        user_id = str(self.user.id)
        project_id = str(self.project.id)
        
        # Generate version
        version = f"v_{int(datetime.utcnow().timestamp())}"
        model_name = f"{self.model_type}_{self.session_id}"
        
        # Prepare metadata
        metadata = {
            "model_type": str(self.model_type),
            "session_id": self.session_id,
            "mlflow_run_id": mlflow_run_id,
            "hyperparameters": results.get('hyperparameters', {}),
            "performance_metrics": results.get('test_metrics', {}),
            "training_date": datetime.utcnow().isoformat(),
            "is_classifier": self.is_classifier
        }
        
        # Version with DVC
        dvc_result = await self.dvc_service.version_model(
            model_path=model_path,
            user_id=user_id,
            project_id=project_id,
            model_name=model_name,
            version=version,
            db_session=self.db_session,
            mongo_db=self.mongo_db,
            metadata=metadata
        )
        
        if dvc_result.get("success"):
            print(f"✅ Model versioned with DVC: {version}")
            self.model_version_id = dvc_result.get("model_version_id")
        else:
            print(f"❌ Failed to version model: {dvc_result.get('error')}")
    
    async def _store_session_data(
        self, 
        results: Dict[str, Any], 
        mlflow_run_id: str, 
        mlflow_experiment_id: str
    ):
        """Store session data in MongoDB"""
        if not self.mongo_db:
            return
        
        session_doc = ModelSessionDocument(
            session_id=self.session_id,
            user_id=str(self.user.id) if self.user else None,
            project_id=str(self.project.id) if self.project else None,
            model_type=str(self.model_type),
            is_classifier=self.is_classifier,
            hyperparameters=results.get('hyperparameters', {}),
            training_metrics=results.get('test_metrics', {}),
            cross_validation_scores=results.get('cv_scores', []),
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_id=mlflow_experiment_id,
            expires_at=datetime.utcnow() + timedelta(hours=24)  # 24 hour session
        )
        
        # Upsert session document
        await self.mongo_db.model_sessions.update_one(
            {"session_id": self.session_id},
            {"$set": session_doc.dict()},
            upsert=True
        )
    
    async def _store_experiment_data(
        self, 
        results: Dict[str, Any], 
        mlflow_run_id: str, 
        mlflow_experiment_id: str
    ):
        """Store experiment data in PostgreSQL"""
        if not self.db_session or not self.project:
            return
        
        experiment = MLExperiment(
            project_id=self.project.id,
            user_id=self.user.id if self.user else None,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_run_id=mlflow_run_id,
            name=f"Training_{self.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_type=str(self.model_type),
            status="completed",
            hyperparameters=results.get('hyperparameters', {}),
            metrics=results.get('test_metrics', {}),
            dataset_info={
                "target_column": results.get('target_column'),
                "n_samples": results.get('n_samples'),
                "n_features": results.get('n_features')
            }
        )
        
        self.db_session.add(experiment)
        self.db_session.commit()
        self.db_session.refresh(experiment)
        self.ml_experiment = experiment
    
    async def _store_mlflow_run_data(self, run, results: Dict[str, Any]):
        """Store MLflow run metadata in MongoDB"""
        if not self.mongo_db:
            return
        
        run_doc = MLFlowRunDocument(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            project_id=str(self.project.id) if self.project else "unknown",
            user_id=str(self.user.id) if self.user else "unknown",
            session_id=self.session_id,
            status=run.info.status,
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            parameters=results.get('hyperparameters', {}),
            metrics=results.get('test_metrics', {}),
            artifact_uri=run.info.artifact_uri
        )
        
        await self.mongo_db.mlflow_runs.insert_one(run_doc.dict())
    
    async def _log_error(self, error_message: str, mlflow_run_id: Optional[str] = None):
        """Log error to MongoDB"""
        if not self.mongo_db:
            return
        
        error_doc = {
            "session_id": self.session_id,
            "user_id": str(self.user.id) if self.user else None,
            "project_id": str(self.project.id) if self.project else None,
            "error_message": error_message,
            "mlflow_run_id": mlflow_run_id,
            "timestamp": datetime.utcnow()
        }
        
        await self.mongo_db.training_errors.insert_one(error_doc)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def get_session_data(self) -> Optional[Dict[str, Any]]:
        """Get session data from MongoDB"""
        if not self.mongo_db:
            return None
        
        session = await self.mongo_db.model_sessions.find_one(
            {"session_id": self.session_id}
        )
        return session
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        if not self.mongo_db:
            return
        
        await self.mongo_db.model_sessions.delete_many({
            "expires_at": {"$lt": datetime.utcnow()}
        })
