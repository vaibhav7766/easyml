from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle


class Metrics:
    def __init__(
        self,
        df: pd.DataFrame,
        x: list[str],
        y: str,
        model_path: str,
        project_id: int,
    ) -> None:
        self.df = df
        self.x = df[x]
        self.y = df[y]
        self.model_path = model_path
        self.project_id = project_id
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

    def metrics_regression(self, y_pred):
        metric = pd.DataFrame()
        metric["MAE"] = [mean_absolute_error(self.y_test, y_pred)]
        metric["MSE"] = [mean_squared_error(self.y_test, y_pred)]
        metric["RMSE"] = [np.sqrt(mean_squared_error(self.y_test, y_pred))]
        metric["MAPE"] = [mean_absolute_percentage_error(self.y_test, y_pred)]
        metric["R2"] = [r2_score(self.y_test, y_pred)]
        return metric.to_html(index=False)

    def metrics_classification(self, y_pred):
        metric = classification_report(self.y_test, y_pred, output_dict=True)
        metric = pd.DataFrame(metric).transpose()
        metric.reset_index(inplace=True)
        metric.rename(columns={"index": "metric"}, inplace=True)
        return metric.to_html(index=False)

    def liner_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_regression(y_pred)
        with open(
            f"{self.model_path}liner_regression_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_classification(y_pred)
        with open(
            f"{self.model_path}logistic_regression_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_classification(y_pred)
        with open(
            f"{self.model_path}decision_tree_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def support_vector_machine(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_classification(y_pred)
        with open(
            f"{self.model_path}support_vector_machine_model_{self.project_id}.pkl",
            "wb",
        ) as file:
            pickle.dump(model, file)
        return metrics

    def k_nearest_neighbors(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_classification(y_pred)
        with open(
            f"{self.model_path}k_nearest_neighbors_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def polynomial_regression(self):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.X_train)
        model = LinearRegression()
        model.fit(X_poly, self.y_train)
        y_pred = model.predict(poly.fit_transform(self.X_test))
        metrics = self.metrics_regression(y_pred)
        with open(
            f"{self.model_path}polynomial_regression_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def lasso_regression(self):
        model = Lasso()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_regression(y_pred)
        with open(
            f"{self.model_path}lasso_regression_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics

    def ridge_regression(self):
        model = Ridge()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.metrics_regression(y_pred)
        with open(
            f"{self.model_path}ridge_regression_model_{self.project_id}.pkl", "wb"
        ) as file:
            pickle.dump(model, file)
        return metrics
