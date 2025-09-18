# 📁 EasyML `/app/models` and `/app/schemas` Folders Explanation

## 🎯 Overview

The EasyML platform uses a **hybrid database architecture** with PostgreSQL and MongoDB, and employs a clear separation between **data models** (database layer) and **schemas** (API layer). Here's a comprehensive explanation of both folders and their roles.

---

## 📊 `/app/models` - Database Models Layer

The models folder contains the **database layer definitions** for both PostgreSQL (relational) and MongoDB (document) databases. This layer handles data persistence, relationships, and database operations.

### 🗂️ File Structure:
```
app/models/
├── __init__.py                 # Package initialization
├── database.py                 # MongoDB CRUD operations (legacy)
├── mongo_schemas.py            # MongoDB document schemas (Pydantic)
└── sql_models.py               # PostgreSQL models (SQLAlchemy)
```

### 🐘 PostgreSQL Models (`sql_models.py`)
**Purpose:** Defines relational database models using SQLAlchemy ORM

#### Core Models:
1. **User Model**
   ```python
   class User(Base):
       id = Column(UUID, primary_key=True)
       username = Column(String(50), unique=True)
       email = Column(String(100), unique=True)
       hashed_password = Column(String(255))
       # Relationships
       projects = relationship("Project", back_populates="owner")
   ```

2. **Project Model**
   ```python
   class Project(Base):
       id = Column(UUID, primary_key=True)
       name = Column(String(100))
       description = Column(Text)
       owner_id = Column(UUID, ForeignKey("users.id"))
       settings = Column(JSONB)  # PostgreSQL JSON field
   ```

3. **MLExperiment Model**
   - Links to projects and users
   - Stores experiment metadata
   - Tracks MLflow integration

4. **ModelVersion Model**
   - Tracks model versions and deployments
   - Links to DVC storage paths
   - Performance metrics storage

5. **ModelDeployment Model**
   - Deployment configurations
   - Kubernetes metadata
   - Container settings

#### Key Features:
- **UUID Primary Keys:** All models use UUID for distributed system compatibility
- **Relationships:** Proper foreign key relationships between entities
- **JSON Storage:** JSONB columns for flexible metadata
- **Audit Fields:** Created/updated timestamps
- **PostgreSQL Features:** Leverages advanced PostgreSQL features

### 🍃 MongoDB Schemas (`mongo_schemas.py`)
**Purpose:** Document validation schemas for MongoDB using Pydantic

#### Core Document Types:

1. **ModelSessionDocument**
   ```python
   class ModelSessionDocument(BaseModel):
       session_id: str
       user_id: Optional[str]
       project_id: Optional[str]
       model_type: Optional[str]
       training_metrics: Dict[str, Any]
       hyperparameters: Dict[str, Any]
       status: SessionStatus
   ```

2. **MLflowMetadataDocument**
   - MLflow run tracking
   - Experiment metadata
   - Artifact paths

3. **DVCMetadataDocument**
   - DVC version control metadata
   - File checksums and paths
   - Version history

4. **ProjectConfigDocument**
   - Project configuration settings
   - Storage paths
   - MLflow experiment links

#### Key Features:
- **Pydantic Validation:** Type safety and validation
- **Flexible Schema:** Document structure can evolve
- **Session Management:** Training session tracking
- **Metadata Storage:** Rich metadata for ML workflows

### 🔄 MongoDB CRUD Operations (`database.py`)
**Purpose:** Direct MongoDB operations and legacy compatibility

#### Classes:
1. **ProjectModel** - Project CRUD operations
2. **ModelResultModel** - ML results storage
3. **DatasetMetadataModel** - Dataset information

---

## 📋 `/app/schemas` - API Schema Layer

The schemas folder contains **Pydantic models** that define the structure of API requests and responses. These schemas handle validation, serialization, and documentation.

### 🗂️ File Structure:
```
app/schemas/
├── __init__.py                 # Package initialization
├── deployment_schemas.py       # Deployment-specific schemas
└── schemas.py                  # General API schemas
```

### 🌐 General API Schemas (`schemas.py`)
**Purpose:** Request/response models for core API endpoints

#### Core Schema Categories:

1. **Project Schemas**
   ```python
   class ProjectCreate(BaseSchema):
       name: str = Field(..., min_length=1, max_length=255)
       description: Optional[str] = None
       user_id: str
   
   class ProjectResponse(BaseSchema):
       id: str
       name: str
       status: ProjectStatus
       created_at: datetime
   ```

2. **File Upload Schemas**
   - `FileInfoResponse` - File metadata
   - `DataPreviewResponse` - Data preview information
   - `FileUploadResponse` - Upload results

3. **Model Training Schemas**
   ```python
   class ModelTrainingRequest(BaseSchema):
       file_id: str
       target_column: str
       model_type: ModelType
       test_size: float = Field(0.2, ge=0.1, le=0.5)
       hyperparameters: Optional[Dict[str, Any]]
   
   class ModelTrainingResponse(BaseSchema):
       success: bool
       model_type: str
       train_metrics: Dict[str, float]
       test_metrics: Dict[str, float]
   ```

4. **Preprocessing Schemas**
   - Request/response for data preprocessing
   - Validation for preprocessing operations

#### Key Features:
- **Input Validation:** Field validation with constraints
- **Type Safety:** Strong typing for API contracts
- **Documentation:** Auto-generated API docs
- **Enum Integration:** Uses core enums for consistency

### 🚀 Deployment Schemas (`deployment_schemas.py`)
**Purpose:** Complex deployment configuration schemas

#### Key Schemas:

1. **DeploymentRequest & DeploymentResponse**
   - Complete deployment configurations
   - Status tracking and progress

2. **ModelSelectionCriteria**
   ```python
   class ModelSelectionCriteria(BaseModel):
       primary_metric: MetricType
       metric_threshold: float
       secondary_metrics: Dict[MetricType, float]
       max_model_size_mb: Optional[float]
       prefer_latest: bool = True
   ```

3. **ContainerConfig**
   ```python
   class ContainerConfig(BaseModel):
       base_image: str = "python:3.10-slim"
       cpu_limit: str = "1000m"
       memory_limit: str = "2Gi"
       env_vars: Dict[str, str]
   ```

4. **DeploymentConfig**
   - Complete deployment specification
   - Kubernetes configuration
   - Scaling and resource settings

---

## 🔄 Data Flow Architecture

### Database Layer (Models) ↔ API Layer (Schemas)
```
API Request (Pydantic Schema)
        ↓
Business Logic (Services)
        ↓
Database Models (SQLAlchemy/MongoDB)
        ↓
Database Storage (PostgreSQL/MongoDB)
        ↓
Database Models (ORM/ODM)
        ↓
API Response (Pydantic Schema)
```

### Example Flow: Model Training
1. **API Request:** `ModelTrainingRequest` schema validates input
2. **Service Layer:** `EnhancedModelTrainingService` processes training
3. **PostgreSQL:** `MLExperiment` model stores experiment metadata
4. **MongoDB:** `ModelSessionDocument` stores training session data
5. **API Response:** `ModelTrainingResponse` schema returns results

---

## 🎯 Key Design Principles

### 1. **Separation of Concerns**
- **Models:** Database structure and persistence
- **Schemas:** API contracts and validation
- **Services:** Business logic and orchestration

### 2. **Database Specialization**
- **PostgreSQL:** Structured data, relationships, ACID transactions
- **MongoDB:** Flexible documents, ML metadata, session data

### 3. **Type Safety**
- **Pydantic:** Runtime validation and type checking
- **SQLAlchemy:** ORM with type hints
- **Enums:** Consistent value constraints

### 4. **Scalability**
- **UUID:** Distributed system friendly
- **Async Support:** Non-blocking database operations
- **Document Flexibility:** Schema evolution support

---

## 🛠️ Usage Examples

### Creating a New Project (Full Stack)
```python
# 1. API Schema (Request Validation)
project_data = ProjectCreate(
    name="ML Classification Project",
    description="Customer churn prediction",
    user_id="user123"
)

# 2. Service Layer (Business Logic)
project_service = ProjectService()
project = await project_service.create_project(
    db=db_session,
    mongo_db=mongo_database,
    user=current_user,
    name=project_data.name,
    description=project_data.description
)

# 3. Database Models (Persistence)
# PostgreSQL: Project table entry
# MongoDB: ProjectConfigDocument

# 4. API Schema (Response)
return ProjectResponse.from_orm(project)
```

### Model Training Session
```python
# API Request Schema
training_request = ModelTrainingRequest(
    file_id="dataset123",
    target_column="churn",
    model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
    test_size=0.2
)

# MongoDB Document
session_doc = ModelSessionDocument(
    session_id="session123",
    model_type="random_forest_classifier",
    training_metrics={"accuracy": 0.95},
    status=SessionStatus.ACTIVE
)

# PostgreSQL Model
experiment = MLExperiment(
    project_id=project.id,
    user_id=user.id,
    model_type="random_forest_classifier",
    status="completed"
)
```

---

## 📈 Benefits of This Architecture

### ✅ **Advantages:**

1. **Clear Separation:** Models vs Schemas have distinct responsibilities
2. **Type Safety:** Pydantic ensures data validation at API boundaries
3. **Database Optimization:** Each database used for its strengths
4. **Maintainability:** Well-organized, modular code structure
5. **API Documentation:** Auto-generated docs from schemas
6. **Flexibility:** MongoDB allows schema evolution
7. **Consistency:** Shared enums and base classes

### 🔧 **Best Practices:**

1. **Never mix** database models with API schemas
2. **Use services** to orchestrate between layers
3. **Validate early** with Pydantic schemas
4. **Keep models focused** on data persistence
5. **Use relationships** properly in SQLAlchemy
6. **Document schemas** with descriptions and examples

---

This architecture provides a robust, scalable foundation for the EasyML platform, separating concerns while enabling complex ML workflows across multiple databases.
