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
    
    async def train_model_with_persistence(self, data, target_column, model_type, test_size=0.2, 
                                          hyperparameters=None, use_cross_validation=False, 
                                          cv_folds=5, auto_version=True):
        """
        Enhanced model training with automatic persistence, MLflow logging, and DVC versioning
        """
        print("🚀 DEBUG: train_model_with_persistence called!")
        print(f"🔍 DEBUG: auto_version={auto_version}, project_id={getattr(self.project, 'id', 'None')}")
        
        mlflow_run_info = None
        
        try:
            # Call the parent class train_model method to do the actual training
            print("🔍 DEBUG: Calling parent train_model method...")
            results = self.train_model(
                data=data,
                target_column=target_column,
                model_type=model_type,
                test_size=test_size,
                hyperparameters=hyperparameters,
                use_cross_validation=use_cross_validation,
                cv_folds=cv_folds,
                experiment_name=f"project_{self.project.id if self.project else 'unknown'}"
            )
            
            print(f"🔍 DEBUG: Training completed successfully! Model type: {results.get('model_type')}")
            
            # Check what keys are in results
            print(f"🔍 DEBUG: Results keys: {list(results.keys())}")
            
            # Since the parent class doesn't return MLflow IDs, capture them directly
            import mlflow
            active_run = mlflow.active_run()
            
            if active_run:
                mlflow_run_id = active_run.info.run_id
                mlflow_experiment_id = active_run.info.experiment_id
                print(f"🔍 DEBUG: Found active MLflow run - run_id: {mlflow_run_id}, experiment_id: {mlflow_experiment_id}")
            else:
                # Try to get the most recent run from the current experiment
                try:
                    experiment_name = f"project_{self.project.id if self.project else 'unknown'}"
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1, order_by=["start_time desc"])
                        if not runs.empty:
                            mlflow_run_id = runs.iloc[0]['run_id']
                            mlflow_experiment_id = experiment.experiment_id
                            print(f"🔍 DEBUG: Found recent MLflow run - run_id: {mlflow_run_id}, experiment_id: {mlflow_experiment_id}")
                        else:
                            mlflow_run_id = None
                            mlflow_experiment_id = None
                            print(f"🔍 DEBUG: No MLflow runs found in experiment")
                    else:
                        mlflow_run_id = None
                        mlflow_experiment_id = None
                        print(f"🔍 DEBUG: No MLflow experiment found")
                except Exception as e:
                    mlflow_run_id = None
                    mlflow_experiment_id = None
                    print(f"🔍 DEBUG: Error getting MLflow run: {e}")
            
            print(f"🔍 DEBUG: Final MLflow info - run_id: {mlflow_run_id}, experiment_id: {mlflow_experiment_id}")
            
            if mlflow_run_id:
                print(f"🔍 DEBUG: MLflow run ID: {mlflow_run_id}")
                
                # Store experiment data in PostgreSQL
                if self.db_session and self.project:
                    print(f"🔍 DEBUG: Storing experiment data...")
                    await self._store_experiment_data(results, mlflow_run_id, mlflow_experiment_id)
                
                # Check conditions for model saving
                print(f"🔍 DEBUG: Model saving conditions - auto_version={auto_version}, project={bool(self.project)}, model={bool(self.model)}")
                
                # Save model to filesystem if we have a project and model
                model_path = None
                if auto_version and self.project and self.model:
                    print("🔍 DEBUG: Saving model to project...")
                    model_path = await self._save_model_to_project(results, mlflow_run_id)
                    
                    if model_path:
                        print(f"🔍 DEBUG: Model saved to: {model_path}")
                        
                        # Create ModelVersion record in PostgreSQL  
                        await self._create_model_version_record(results, model_path, mlflow_run_id)
                        
                        # Version with DVC
                        await self._version_model_with_dvc(model_path, results, mlflow_run_id)
                    else:
                        print("⚠️ DEBUG: Failed to save model to filesystem")
                
                # Store training session in MongoDB
                if self.mongo_db is not None:
                    await self._store_session_data(results, mlflow_run_id, mlflow_experiment_id)
                
                # Log MLflow run details
                await self._log_mlflow_run(results, mlflow_run_id, mlflow_experiment_id)
            
            print("✅ DEBUG: Training with persistence completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ DEBUG: Training with persistence failed: {str(e)}")
            error_msg = f"Training failed: {str(e)}"
            await self._log_error(error_msg, mlflow_run_id)
            return {"error": error_msg}
    
    async def _save_model_to_project(self, results: Dict[str, Any], mlflow_run_id: str) -> Optional[str]:
        """Save model to project folder structure"""
        print(f"🔍 _save_model_to_project called - project: {bool(self.project)}, model: {bool(self.model)}")
        
        if not self.project or not self.model:
            print(f"⚠️ Cannot save model - project: {bool(self.project)}, model: {bool(self.model)}")
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
        if self.mongo_db is None:
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
        self.mongo_db.model_sessions.update_one(
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
    
    async def _create_model_version_record(
        self, 
        results: Dict[str, Any], 
        model_path: str, 
        mlflow_run_id: str
    ):
        """Create ModelVersion record for deployment service"""
        print(f"🔍 DEBUG: _create_model_version_record called")
        print(f"🔍 DEBUG: db_session={bool(self.db_session)}, project={bool(self.project)}, user={bool(self.user)}")
        print(f"🔍 DEBUG: model_path={model_path}")
        
        if not self.db_session or not self.project or not self.user:
            print(f"⚠️ Skipping ModelVersion creation: db_session={bool(self.db_session)}, project={bool(self.project)}, user={bool(self.user)}")
            return
        
        try:
            # Generate model name
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            model_name = f"{self.model_type}_{timestamp}_{self.session_id[:8]}"
            
            print(f"🔄 DEBUG: Creating ModelVersion record: {model_name}")
            print(f"🔍 DEBUG: project_id={self.project.id}, experiment_id={self.ml_experiment.id if self.ml_experiment else None}")
            
            # Create ModelVersion record with DVC integration
            model_version = ModelVersion(
                project_id=self.project.id,
                experiment_id=self.ml_experiment.id if self.ml_experiment else None,
                name=model_name,
                version="1.0.0",
                model_type=str(self.model_type),
                storage_path=model_path,
                dvc_path=f"{model_path}.dvc",  # DVC file path
                mlflow_run_id=mlflow_run_id,
                mlflow_model_uri=f"runs:/{mlflow_run_id}/model" if mlflow_run_id else None,
                performance_metrics=results.get('test_metrics', {}),
                status="active"
            )
            
            print(f"🔍 DEBUG: ModelVersion object created, adding to session...")
            self.db_session.add(model_version)
            
            print(f"🔍 DEBUG: Committing to database...")
            self.db_session.commit()
            
            print(f"🔍 DEBUG: Refreshing model_version...")
            self.db_session.refresh(model_version)
            self.model_version = model_version
            
            print(f"✅ DEBUG: ModelVersion record created successfully!")
            print(f"✅ DEBUG: ID={model_version.id}, name={model_version.name}, status={model_version.status}")
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in _create_model_version_record: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
            if self.db_session:
                print(f"🔄 DEBUG: Rolling back database transaction...")
                self.db_session.rollback()
            raise e
    
    async def _store_mlflow_run_data(self, run, results: Dict[str, Any]):
        """Store MLflow run metadata in MongoDB"""
        if self.mongo_db is None:
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
        
        self.mongo_db.mlflow_runs.insert_one(run_doc.dict())
    
    async def _log_error(self, error_message: str, mlflow_run_id: Optional[str] = None):
        """Log error to MongoDB"""
        if self.mongo_db is None:
            return
        
        error_doc = {
            "session_id": self.session_id,
            "user_id": str(self.user.id) if self.user else None,
            "project_id": str(self.project.id) if self.project else None,
            "error_message": error_message,
            "mlflow_run_id": mlflow_run_id,
            "timestamp": datetime.utcnow()
        }
        
        self.mongo_db.training_errors.insert_one(error_doc)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def get_session_data(self) -> Optional[Dict[str, Any]]:
        """Get session data from MongoDB"""
        if self.mongo_db is None:
            return None
        
        session = self.mongo_db.model_sessions.find_one(
            {"session_id": self.session_id}
        )
        return session
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        if self.mongo_db is None:
            return
        
        self.mongo_db.model_sessions.delete_many({
            "expires_at": {"$lt": datetime.utcnow()}
        })
    
    async def _log_mlflow_run(self, results: Dict[str, Any], mlflow_run_id: str, mlflow_experiment_id: str):
        """Log MLflow run details for debugging"""
        print(f"🔍 DEBUG: MLflow run logged - run_id: {mlflow_run_id}, experiment_id: {mlflow_experiment_id}")
        print(f"🔍 DEBUG: Training results logged with {len(results)} metrics")
