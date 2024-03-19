import pandas as pd
from sklearn.impute import KNNImputer


class Preprocessing:
    df = None

    def remove_nulls(self, option: str) -> None:
        if option == "median":
            self.df = self.df.fillna(self.df.median())
        elif option == "knn":
            input_neighbours = int(input("Enter the number of neighbours: "))
            imputer = KNNImputer(n_neighbors=input_neighbours)
            self.df = imputer.fit_transform(self.df)
        elif option == "mean":
            self.df = self.df.fillna(self.df.mean())
