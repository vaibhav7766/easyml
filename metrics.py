
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
# from sklearn.model_selection import GridSearchCV

class Metrics:
    def __init__(self, df: pd.DataFrame,x : list | None, y : list, modal_path: str) -> None:
        self.df = df
        self.x = df[x]
        self.y = df[y]
        self.modal_path = modal_path
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
    def metrics_regression(self,y_pred):
        metric = pd.DataFrame()
        metric["MAE"] = [mean_absolute_error(self.y_test, y_pred)]
        metric["MSE"] = [mean_squared_error(self.y_test, y_pred)]
        metric["RMSE"] = [np.sqrt(mean_squared_error(self.y_test, y_pred))]
        metric["MAPE"] = [mean_absolute_percentage_error(self.y_test, y_pred)]
        metric["R2"] = [r2_score(self.y_test, y_pred)]
        metric["Accuracy"] = [accuracy_score(self.y_test, y_pred)]
        metric["Precision"] = [precision_score(self.y_test, y_pred)]
        metric["Recall"] = [recall_score(self.y_test, y_pred)]
        metric["F1"] = [f1_score(self.y_test, y_pred)]
        metric["True Positive"] = [confusion_matrix(self.y_test, y_pred)[0][0]]
        metric["False Positive"] = [confusion_matrix(self.y_test, y_pred)[0][1]]
        metric["True Negative"] = [confusion_matrix(self.y_test, y_pred)[1][0]]
        metric["False Negative"] = [confusion_matrix(self.y_test, y_pred)[1][1]]
        return metric.to_html()
    
    def metrics_classification(self,y_pred):
        metric = pd.DataFrame()
        metric["Accuracy"] = [accuracy_score(self.y_test, y_pred)]
        metric["Precision"] = [precision_score(self.y_test, y_pred)]
        metric["Recall"] = [recall_score(self.y_test, y_pred)]
        metric["F1"] = [f1_score(self.y_test, y_pred)]
        metric["True Positive"] = [confusion_matrix(self.y_test, y_pred)[0][0]]
        metric["False Positive"] = [confusion_matrix(self.y_test, y_pred)[0][1]]
        metric["True Negative"] = [confusion_matrix(self.y_test, y_pred)[1][0]]
        metric["False Negative"] = [confusion_matrix(self.y_test, y_pred)[1][1]]
        metric["Report"] = [classification_report(self.y_test, y_pred)]
        return metric.to_html()
        
    def liner_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_regression(self,y_pred)
        # save modal
        
        return metrics
    
    def logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_classification(self,y_pred)
        return metrics

    def decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_classification(self,y_pred)
        return metrics
    
    def support_vector_machine(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_classification(self,y_pred)
        return metrics

    
    def k_nearest_neighbors(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_classification(self,y_pred)
        return metrics
    
    
    def polynomial_regression(self):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.X_train)
        model = LinearRegression()
        model.fit(X_poly, self.y_train)
        y_pred = model.predict(poly.fit_transform(self.X_test))
        metrics = Metrics.metrics_regression(self,y_pred)
        return metrics
    
    def lasso_regression(self):
        model = Lasso()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_regression(self,y_pred)
        return metrics
    
    def ridge_regression(self):
        model = Ridge()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = Metrics.metrics_regression(self,y_pred)
        return metrics
    
    