import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import json
import joblib
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parameters():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['deployment']
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_model_artifacts(model_dir):
    """Load all model artifacts"""
    try:
        artifacts = {}
        
        # Load model
        model_path = os.path.join(model_dir, 'model.joblib')
        if os.path.exists(model_path):
            artifacts['model'] = joblib.load(model_path)
            logger.info("Model loaded successfully")
        
        # Load preprocessors
        preprocessor_dir = model_dir.replace('models', 'features')
        
        scaler_path = os.path.join(preprocessor_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            artifacts['scaler'] = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        
        encoders_path = os.path.join(preprocessor_dir, 'encoders.joblib')
        if os.path.exists(encoders_path):
            artifacts['encoders'] = joblib.load(encoders_path)
            logger.info("Encoders loaded successfully")
        
        selector_path = os.path.join(preprocessor_dir, 'feature_selector.joblib')
        if os.path.exists(selector_path):
            artifacts['feature_selector'] = joblib.load(selector_path)
            logger.info("Feature selector loaded successfully")
        
        # Load metadata
        info_path = os.path.join(model_dir, 'training_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                artifacts['training_info'] = json.load(f)
            logger.info("Training info loaded successfully")
        
        features_path = os.path.join(preprocessor_dir, 'selected_features.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                artifacts['selected_features'] = json.load(f)
            logger.info("Selected features loaded successfully")
        
        return artifacts
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise

def validate_model_for_deployment(artifacts, params):
    """Validate model meets deployment criteria"""
    logger.info("Validating model for deployment...")
    
    validation_results = {}
    
    # Check required artifacts
    required_artifacts = ['model', 'training_info']
    for artifact in required_artifacts:
        if artifact not in artifacts:
            logger.error(f"Missing required artifact: {artifact}")
            validation_results[f"missing_{artifact}"] = True
        else:
            validation_results[f"has_{artifact}"] = True
    
    # Load evaluation results for performance check
    model_dir = None
    if 'training_info' in artifacts:
        # Try to find evaluation results
        eval_dir = 'evaluation'
        eval_path = os.path.join(eval_dir, 'evaluation_results.json')
        
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            
            metrics = eval_results.get('metrics', {})
            selection_criteria = params.get('model_selection_criteria', {})
            
            primary_metric = selection_criteria.get('primary_metric', 'f1_score')
            
            if primary_metric in metrics:
                validation_results['primary_metric_value'] = metrics[primary_metric]
                validation_results['primary_metric_name'] = primary_metric
                logger.info(f"Primary metric ({primary_metric}): {metrics[primary_metric]:.4f}")
            
            # Check secondary metrics
            secondary_metrics = selection_criteria.get('secondary_metrics', [])
            validation_results['secondary_metrics'] = {}
            for metric in secondary_metrics:
                if metric in metrics:
                    validation_results['secondary_metrics'][metric] = metrics[metric]
                    logger.info(f"Secondary metric ({metric}): {metrics[metric]:.4f}")
    
    # Model size check
    model_size_limit = params.get('model_selection_criteria', {}).get('model_size_limit_mb', 100)
    if 'model' in artifacts:
        # Estimate model size
        import pickle
        model_bytes = pickle.dumps(artifacts['model'])
        model_size_mb = len(model_bytes) / (1024 * 1024)
        
        validation_results['model_size_mb'] = model_size_mb
        validation_results['size_within_limit'] = model_size_mb <= model_size_limit
        
        if model_size_mb <= model_size_limit:
            logger.info(f"Model size: {model_size_mb:.2f} MB (within limit of {model_size_limit} MB)")
        else:
            logger.warning(f"Model size: {model_size_mb:.2f} MB (exceeds limit of {model_size_limit} MB)")
    
    return validation_results

def optimize_model(artifacts, params):
    """Optimize model for deployment"""
    logger.info("Optimizing model for deployment...")
    
    optimization_config = params.get('optimization', {})
    optimization_level = params.get('optimization_level', 'moderate')
    
    optimized_artifacts = artifacts.copy()
    optimization_results = {
        'optimization_level': optimization_level,
        'optimizations_applied': []
    }
    
    # ONNX conversion
    if optimization_config.get('onnx_conversion', False):
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            model = artifacts['model']
            training_info = artifacts.get('training_info', {})
            
            # Determine input shape
            feature_names = training_info.get('feature_names', [])
            input_shape = (None, len(feature_names))
            
            initial_type = [('float_input', FloatTensorType(input_shape))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            optimized_artifacts['onnx_model'] = onnx_model
            optimization_results['optimizations_applied'].append('onnx_conversion')
            
            logger.info("Model converted to ONNX format")
            
        except ImportError:
            logger.warning("skl2onnx not available, skipping ONNX conversion")
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
    
    # Model compression (if applicable)
    if optimization_level in ['moderate', 'aggressive']:
        # For tree-based models, we could implement pruning
        # For neural networks, we could implement quantization
        # This is a placeholder for actual optimization logic
        optimization_results['optimizations_applied'].append('compression_attempted')
        logger.info("Model compression attempted")
    
    return optimized_artifacts, optimization_results

def create_deployment_package(artifacts, output_dir, params):
    """Create deployment package"""
    logger.info("Creating deployment package...")
    
    # Create deployment directory structure
    deployment_dirs = [
        'models',
        'preprocessors',
        'metadata',
        'config'
    ]
    
    for dir_name in deployment_dirs:
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'models', 'model.joblib')
    joblib.dump(artifacts['model'], model_path)
    
    # Save ONNX model if available
    if 'onnx_model' in artifacts:
        onnx_path = os.path.join(output_dir, 'models', 'model.onnx')
        with open(onnx_path, 'wb') as f:
            f.write(artifacts['onnx_model'].SerializeToString())
    
    # Save preprocessors
    for preprocessor_name in ['scaler', 'encoders', 'feature_selector']:
        if preprocessor_name in artifacts:
            preprocessor_path = os.path.join(output_dir, 'preprocessors', f'{preprocessor_name}.joblib')
            joblib.dump(artifacts[preprocessor_name], preprocessor_path)
    
    # Save metadata
    metadata_files = ['training_info', 'selected_features']
    for metadata_name in metadata_files:
        if metadata_name in artifacts:
            metadata_path = os.path.join(output_dir, 'metadata', f'{metadata_name}.json')
            with open(metadata_path, 'w') as f:
                json.dump(artifacts[metadata_name], f, indent=2, default=str)
    
    # Create API configuration
    api_config = {
        'model_type': artifacts.get('training_info', {}).get('model_type', 'unknown'),
        'task_type': artifacts.get('training_info', {}).get('task_type', 'unknown'),
        'feature_names': artifacts.get('selected_features', []),
        'api_settings': params.get('api', {}),
        'deployment_timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    config_path = os.path.join(output_dir, 'config', 'api_config.json')
    with open(config_path, 'w') as f:
        json.dump(api_config, f, indent=2)
    
    # Create inference script
    inference_script = '''
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ModelInference:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all model artifacts"""
        # Load model
        model_path = os.path.join(self.model_dir, 'models', 'model.joblib')
        self.model = joblib.load(model_path)
        
        # Load metadata
        with open(os.path.join(self.model_dir, 'metadata', 'training_info.json'), 'r') as f:
            self.training_info = json.load(f)
        
        with open(os.path.join(self.model_dir, 'config', 'api_config.json'), 'r') as f:
            self.api_config = json.load(f)
        
        # Load preprocessors
        self.preprocessors = {}
        for preprocessor in ['scaler', 'encoders', 'feature_selector']:
            preprocessor_path = os.path.join(self.model_dir, 'preprocessors', f'{preprocessor}.joblib')
            if os.path.exists(preprocessor_path):
                self.preprocessors[preprocessor] = joblib.load(preprocessor_path)
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        processed_data = data.copy()
        
        # Apply encoders
        if 'encoders' in self.preprocessors:
            encoders = self.preprocessors['encoders']
            for col, encoder in encoders.items():
                if col in processed_data.columns:
                    try:
                        processed_data[col] = encoder.transform(processed_data[[col]])
                    except:
                        # Handle unknown categories
                        pass
        
        # Apply scaler
        if 'scaler' in self.preprocessors:
            scaler = self.preprocessors['scaler']
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = scaler.transform(processed_data[numeric_cols])
        
        # Apply feature selection
        if 'feature_selector' in self.preprocessors:
            selector = self.preprocessors['feature_selector']
            processed_data = pd.DataFrame(
                selector.transform(processed_data),
                columns=self.api_config['feature_names']
            )
        
        return processed_data
    
    def predict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions"""
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df = self.preprocess(df)
        
        # Make predictions
        predictions = self.model.predict(processed_df)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': pred,
                'input_data': data[i]
            }
            
            # Add prediction probabilities for classification
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(processed_df.iloc[[i]])
                    result['probabilities'] = probabilities[0].tolist()
                except:
                    pass
            
            results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize inference
    inference = ModelInference('/path/to/deployment/package')
    
    # Make prediction
    sample_data = [{"feature1": 1.0, "feature2": 2.0}]  # Replace with actual features
    predictions = inference.predict(sample_data)
    print(predictions)
'''
    
    script_path = os.path.join(output_dir, 'inference.py')
    with open(script_path, 'w') as f:
        f.write(inference_script)
    
    # Create README
    readme_content = f'''
# Model Deployment Package

This package contains all artifacts needed to deploy the trained model.

## Contents

- `models/`: Contains the trained model files
- `preprocessors/`: Contains preprocessing artifacts (scalers, encoders, etc.)
- `metadata/`: Contains model metadata and feature information
- `config/`: Contains API configuration
- `inference.py`: Ready-to-use inference script

## Model Information

- Model Type: {api_config['model_type']}
- Task Type: {api_config['task_type']}
- Features: {len(api_config['feature_names'])}
- Deployment Date: {api_config['deployment_timestamp']}

## Usage

```python
from inference import ModelInference

# Initialize
inference = ModelInference('/path/to/this/package')

# Make prediction
data = [{{"feature1": 1.0, "feature2": 2.0}}]  # Replace with actual features
predictions = inference.predict(data)
```

## API Integration

This package is ready for integration with the EasyML API. Copy this package to your deployment environment and update the API endpoints to use the ModelInference class.
'''
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Deployment package created at {output_dir}")
    
    return {
        'package_path': output_dir,
        'model_files': os.listdir(os.path.join(output_dir, 'models')),
        'preprocessor_files': os.listdir(os.path.join(output_dir, 'preprocessors')),
        'metadata_files': os.listdir(os.path.join(output_dir, 'metadata'))
    }

def prepare_deployment(model_dir, output_dir):
    """Main deployment preparation function"""
    logger.info("Starting deployment preparation...")
    
    # Load parameters
    params = load_parameters()
    
    # Load model artifacts
    artifacts = load_model_artifacts(model_dir)
    
    # Validate model for deployment
    validation_results = validate_model_for_deployment(artifacts, params)
    
    # Check if model passes validation
    has_model = validation_results.get('has_model', False)
    has_training_info = validation_results.get('has_training_info', False)
    size_within_limit = validation_results.get('size_within_limit', True)
    
    if not (has_model and has_training_info and size_within_limit):
        logger.error("Model validation failed, cannot proceed with deployment")
        return None
    
    # Optimize model
    optimized_artifacts, optimization_results = optimize_model(artifacts, params)
    
    # Create deployment package
    package_info = create_deployment_package(optimized_artifacts, output_dir, params)
    
    # Save deployment metadata
    deployment_metadata = {
        'deployment_timestamp': datetime.now().isoformat(),
        'model_validation': validation_results,
        'optimization_results': optimization_results,
        'package_info': package_info,
        'deployment_config': params
    }
    
    metadata_path = os.path.join(output_dir, 'deployment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(deployment_metadata, f, indent=2, default=str)
    
    logger.info("Deployment preparation completed successfully")
    logger.info(f"Deployment package available at: {output_dir}")
    
    return deployment_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deployment Preparation Pipeline')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model')
    parser.add_argument('--output', required=True, help='Output directory for deployment package')
    
    args = parser.parse_args()
    
    try:
        prepare_deployment(args.model_dir, args.output)
        logger.info("Deployment preparation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Deployment preparation pipeline failed: {e}")
        sys.exit(1)
