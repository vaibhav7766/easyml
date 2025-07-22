from sklearn.impute import KNNImputer
import pandas as pd

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.is_categorical = False  # Default value, can be set externally
    
    def encode(self, choice: str, column: str):
        """Handle categorical encoding in the dataset."""
        if choice == "one-hot":
            self.data = pd.get_dummies(self.data, columns=[column], drop_first=True)
        elif choice == "label":
            self.data[column] = self.data[column].astype('category').cat.codes
        return self.data
    
    def imputation(self, choice: str, column: str):
        """Handle missing values in the dataset."""
        # methods = ["mean", "median", "mode", "drop", "knn"]
        if self.data[column].isnull().any():
            if choice == "mean" and self.is_categorical is False :
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif choice == "median":
                self.data[column].fillna(self.data[column].median(), inplace=True)
            elif choice == "mode":
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            elif choice == "drop":
                self.data.dropna(subset=[column], inplace=True)
            elif choice == "knn":
                imputer = KNNImputer(n_neighbors=5)
                self.data[[column]] = imputer.fit_transform(self.data[[column]])
                
        return self.data

    def delete_columns(self, selected_columns: list):
        """Select columns to delete from dataset via drop-down interface."""
        self.data = self.data.drop(columns=selected_columns, errors='ignore')
        return self.data

    def normalize(self, choice: str, column: str):
        """Normalize the dataset based on the choice."""
        if choice == "min-max":
            self.data[column] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
        elif choice == "z-score":
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        elif choice == "max-abs":
            self.data[column] = self.data[column] / self.data[column].abs().max()
        elif choice == "robust":
            median = self.data[column].median()
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1
            self.data[column] = (self.data[column] - median) / iqr
            
        return self.data