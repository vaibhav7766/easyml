"""
Preprocessing service for data cleaning and transformation
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Dict, Any, List, Optional

from app.core.enums import PreprocessingOption


class PreprocessingService:
    """Service for data preprocessing operations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_data = data.copy()
        self.preprocessing_history: List[Dict[str, Any]] = []
    
    def apply_preprocessing(self, choices: Dict[str, str], is_categorical: bool = False) -> Dict[str, Any]:
        """
        Apply preprocessing choices to the dataset
        
        Args:
            choices: Dictionary mapping operation to column
            is_categorical: Whether to treat columns as categorical
            
        Returns:
            Dictionary with processing results and metadata
        """
        results = {
            "applied_operations": [],
            "errors": [],
            "data_shape_before": self.data.shape,
            "data_shape_after": None,
            "columns_before": list(self.data.columns),
            "columns_after": None
        }
        
        for operation, column in choices.items():
            try:
                if operation == "imputation":
                    self._handle_imputation(column, "mean", is_categorical)
                    results["applied_operations"].append(f"Imputation on {column}")
                
                elif operation == "encoding":
                    self._handle_encoding(column, "one-hot")
                    results["applied_operations"].append(f"Encoding on {column}")
                
                elif operation == "normalization":
                    self._handle_normalization(column, "z-score")
                    results["applied_operations"].append(f"Normalization on {column}")
                
                elif operation == "cleaning":
                    self._handle_cleaning(column)
                    results["applied_operations"].append(f"Cleaning on {column}")
                
                else:
                    results["errors"].append(f"Unknown operation: {operation}")
            
            except Exception as e:
                results["errors"].append(f"Error processing {operation} on {column}: {str(e)}")
        
        results["data_shape_after"] = self.data.shape
        results["columns_after"] = list(self.data.columns)
        
        return results
    
    def _handle_imputation(self, column: str, method: str, is_categorical: bool = False):
        """Handle missing values in the dataset"""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        if not self.data[column].isnull().any():
            return  # No missing values to handle
        
        if method == "mean" and not is_categorical:
            self.data[column].fillna(self.data[column].mean(), inplace=True)
        elif method == "median":
            self.data[column].fillna(self.data[column].median(), inplace=True)
        elif method == "mode":
            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        elif method == "drop":
            self.data.dropna(subset=[column], inplace=True)
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            self.data[[column]] = imputer.fit_transform(self.data[[column]])
        elif method == "forward_fill":
            self.data[column].fillna(method='ffill', inplace=True)
        elif method == "backward_fill":
            self.data[column].fillna(method='bfill', inplace=True)
        else:
            # Default to most frequent for categorical, mean for numeric
            if is_categorical or self.data[column].dtype == 'object':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            else:
                self.data[column].fillna(self.data[column].mean(), inplace=True)
    
    def _handle_encoding(self, column: str, method: str):
        """Handle categorical encoding"""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        if method == "one-hot":
            # One-hot encoding
            encoded = pd.get_dummies(self.data[column], prefix=column, drop_first=True)
            self.data = pd.concat([self.data.drop(columns=[column]), encoded], axis=1)
        
        elif method == "label":
            # Label encoding
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column].astype(str))
        
        elif method == "ordinal":
            # Simple ordinal encoding (categorical codes)
            self.data[column] = self.data[column].astype('category').cat.codes
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def _handle_normalization(self, column: str, method: str):
        """Normalize numerical columns"""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        if self.data[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column {column} is not numeric")
        
        if method == "min-max":
            scaler = MinMaxScaler()
            self.data[column] = scaler.fit_transform(self.data[[column]]).flatten()
        
        elif method == "z-score" or method == "standard":
            scaler = StandardScaler()
            self.data[column] = scaler.fit_transform(self.data[[column]]).flatten()
        
        elif method == "robust":
            scaler = RobustScaler()
            self.data[column] = scaler.fit_transform(self.data[[column]]).flatten()
        
        elif method == "max-abs":
            max_abs_value = self.data[column].abs().max()
            if max_abs_value != 0:
                self.data[column] = self.data[column] / max_abs_value
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _handle_cleaning(self, column: str):
        """Handle basic data cleaning"""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        # Remove duplicate rows
        initial_shape = self.data.shape
        self.data.drop_duplicates(inplace=True)
        
        # For string columns, clean whitespace
        if self.data[column].dtype == 'object':
            self.data[column] = self.data[column].astype(str).str.strip()
            # Remove empty strings
            self.data = self.data[self.data[column] != '']
        
        # For numeric columns, remove extreme outliers (beyond 3 std)
        elif self.data[column].dtype in ['int64', 'float64']:
            mean = self.data[column].mean()
            std = self.data[column].std()
            self.data = self.data[
                (self.data[column] >= mean - 3*std) & 
                (self.data[column] <= mean + 3*std)
            ]
    
    def delete_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Delete specified columns from dataset"""
        result = {
            "deleted_columns": [],
            "not_found_columns": [],
            "shape_before": self.data.shape,
            "shape_after": None
        }
        
        for column in columns:
            if column in self.data.columns:
                self.data.drop(columns=[column], inplace=True)
                result["deleted_columns"].append(column)
            else:
                result["not_found_columns"].append(column)
        
        result["shape_after"] = self.data.shape
        return result
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the current dataset"""
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.data.select_dtypes(include=['object', 'category']).columns),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "duplicate_rows": self.data.duplicated().sum()
        }
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get the processed dataset"""
        return self.data.copy()
    
    def reset_data(self):
        """Reset data to original state"""
        self.data = self.original_data.copy()
        self.preprocessing_history.clear()
