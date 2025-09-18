# Dataset Versioning System

## Overview

The EasyML platform now includes an automatic dataset versioning system that tracks all dataset changes throughout the data processing pipeline. This system uses semantic versioning to maintain a clear history of dataset evolution.

## Versioning Strategy

### Version Numbering
- **V1**: Initial raw data upload
- **V2**: First preprocessing operation
- **V2.1, V2.2, V2.3...**: Subsequent preprocessing operations

### Version Tags
- **"raw data"**: Original uploaded datasets (V1)
- **"preprocessed data"**: Any dataset that has undergone preprocessing (V2+)

## Database Schema

### DatasetVersion Table
```sql
CREATE TABLE dataset_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id),
    name VARCHAR(200) NOT NULL,           -- Dataset name (filename without extension)
    version VARCHAR(50) NOT NULL,         -- Version number (V1, V2, V2.1, etc.)
    tag VARCHAR(100) NOT NULL DEFAULT 'raw data',  -- 'raw data' or 'preprocessed data'
    storage_path VARCHAR(500) NOT NULL,   -- File system path
    dvc_path VARCHAR(500),               -- DVC path (optional)
    size_bytes INTEGER,                  -- File size
    num_rows INTEGER,                    -- Number of rows
    num_columns INTEGER,                 -- Number of columns
    checksum VARCHAR(100),               -- File checksum
    schema_info JSONB DEFAULT '{}',      -- Column types and schema
    statistics JSONB DEFAULT '{}',       -- Dataset statistics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## API Endpoints

### 1. Upload Dataset (Creates V1)
```http
POST /upload/?project_id={project_id}
Content-Type: multipart/form-data

file: [dataset file]
```

**Response:**
```json
{
    "success": true,
    "file_id": "uuid",
    "original_filename": "dataset.csv",
    "file_info": {
        "dataset_version": "V1",
        "dataset_tag": "raw data",
        "dataset_version_id": "uuid",
        "num_rows": 1000,
        "num_columns": 5
    },
    "message": "File uploaded successfully as V1 (raw data)",
    "project_id": "project_uuid"
}
```

### 2. Apply Preprocessing (Creates V2, V2.1, etc.)
```http
POST /preprocessing/apply?project_id={project_id}
Content-Type: application/json

{
    "file_id": "uuid",
    "operations": {
        "standardize": ["age", "salary"],
        "remove_duplicates": true
    },
    "is_categorical": {}
}
```

**Response:**
```json
{
    "success": true,
    "applied_operations": ["standardize", "remove_duplicates"],
    "errors": [],
    "data_shape_before": [1000, 5],
    "data_shape_after": [995, 5],
    "columns_before": ["name", "age", "salary", "department", "city"],
    "columns_after": ["name", "age", "salary", "department", "city"],
    "data_summary": {...},
    "message": "Preprocessing applied successfully. Created V2 (preprocessed data)"
}
```

### 3. List Dataset Versions
```http
GET /preprocessing/dataset-versions/{project_id}?dataset_name={optional}
```

**Response:**
```json
{
    "success": true,
    "project_id": "project_uuid",
    "dataset_name": "employee_data",
    "versions": [
        {
            "id": "uuid",
            "name": "employee_data",
            "version": "V2.1",
            "tag": "preprocessed data",
            "storage_path": "/path/to/preprocessed_employee_data_V2.1.csv",
            "size_bytes": 45678,
            "num_rows": 995,
            "num_columns": 5,
            "created_at": "2025-09-18T10:30:00Z",
            "schema_info": {"name": "object", "age": "int64", ...},
            "statistics": {"mean": {...}, "std": {...}, "null_counts": {...}}
        },
        {
            "id": "uuid",
            "name": "employee_data",
            "version": "V2",
            "tag": "preprocessed data",
            "storage_path": "/path/to/preprocessed_employee_data_V2.csv",
            "size_bytes": 46234,
            "num_rows": 998,
            "num_columns": 5,
            "created_at": "2025-09-18T10:15:00Z"
        },
        {
            "id": "uuid",
            "name": "employee_data",
            "version": "V1",
            "tag": "raw data",
            "storage_path": "/path/to/employee_data.csv",
            "size_bytes": 50000,
            "num_rows": 1000,
            "num_columns": 5,
            "created_at": "2025-09-18T10:00:00Z"
        }
    ],
    "total_versions": 3
}
```

## Workflow Examples

### Typical Data Processing Workflow

1. **Initial Upload**
   ```
   User uploads "sales_data.csv" → Creates V1 (raw data)
   ```

2. **First Preprocessing**
   ```
   Apply data cleaning → Creates V2 (preprocessed data)
   ```

3. **Additional Preprocessing**
   ```
   Apply feature engineering → Creates V2.1 (preprocessed data)
   Apply normalization → Creates V2.2 (preprocessed data)
   ```

4. **Back to Preprocessing**
   ```
   User goes back to preprocessing → Creates V2.3 (preprocessed data)
   ```

### Version Evolution Timeline
```
V1 (raw data) → V2 (preprocessed) → V2.1 (preprocessed) → V2.2 (preprocessed)
     ↑               ↑                    ↑                      ↑
 Initial Upload   First Process    Second Process         Third Process
```

## Benefits

1. **Complete Audit Trail**: Track every change made to datasets
2. **Reproducibility**: Know exactly which version was used for training
3. **Rollback Capability**: Return to previous dataset versions if needed
4. **Clear Lineage**: Understand the evolution of your data
5. **Team Collaboration**: Share specific dataset versions with team members

## Implementation Details

### File Storage
- Each version is stored as a separate file
- Files are named with version suffixes: `dataset_V2.1.csv`
- Original files are preserved unchanged

### Version Logic
```python
def get_next_version(current_version):
    if not current_version:
        return "V1"
    elif current_version == "V1":
        return "V2"
    elif "." not in current_version:
        return f"{current_version}.1"
    else:
        major, minor = current_version.rsplit(".", 1)
        return f"{major}.{int(minor) + 1}"
```

### Database Migration
For existing installations, run:
```bash
python scripts/add_dataset_tag_column.py
```

## Testing

Use the provided test script to verify functionality:
```bash
python scripts/test_dataset_versioning.py
```

## Future Enhancements

1. **Version Comparison**: Compare different dataset versions
2. **Branch Versioning**: Support parallel preprocessing paths
3. **Version Merging**: Combine features from different versions
4. **Automatic Cleanup**: Archive old versions based on retention policies
5. **Version Annotations**: Add user comments to versions