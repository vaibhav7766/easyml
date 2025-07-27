// MongoDB initialization script
// Creates collections and indexes for EasyML

db = db.getSiblingDB('easyml');

// Create collections with basic indexes
db.createCollection('model_sessions');
db.model_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.model_sessions.createIndex({ "user_id": 1 });
db.model_sessions.createIndex({ "project_id": 1 });
db.model_sessions.createIndex({ "expires_at": 1 });

db.createCollection('dvc_metadata');
db.dvc_metadata.createIndex({ "file_path": 1 }, { unique: true });
db.dvc_metadata.createIndex({ "project_id": 1 });
db.dvc_metadata.createIndex({ "md5_hash": 1 });

db.createCollection('mlflow_runs');
db.mlflow_runs.createIndex({ "run_id": 1 }, { unique: true });
db.mlflow_runs.createIndex({ "experiment_id": 1 });
db.mlflow_runs.createIndex({ "project_id": 1 });

db.createCollection('project_configs');
db.project_configs.createIndex({ "project_id": 1 }, { unique: true });
db.project_configs.createIndex({ "user_id": 1 });

db.createCollection('audit_logs');
db.audit_logs.createIndex({ "timestamp": -1 });
db.audit_logs.createIndex({ "user_id": 1 });
db.audit_logs.createIndex({ "project_id": 1 });

db.createCollection('training_errors');
db.training_errors.createIndex({ "timestamp": -1 });
db.training_errors.createIndex({ "session_id": 1 });

print('MongoDB collections and indexes created successfully!');
