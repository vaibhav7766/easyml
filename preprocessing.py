import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessing:

    def __init__(
        self, df: pd.DataFrame, csv_name: str, final_csv_path: str, edit_csv_path: str
    ) -> None:
        self.df = df
        self.df_categorical = self.df.select_dtypes("object")
        self.df_numerical = self.df.select_dtypes(["number"])
        self.df_final = self.df
        self.csv_name = csv_name
        self.final_csv_path = final_csv_path
        self.edit_csv_path = edit_csv_path

    def remove_nulls(self, option: str) -> None:
        if option == "median":
            self.df_numerical = self.df_numerical.fillna(self.df_numerical.median())

        elif option == "mean":
            self.df_numerical = self.df_numerical.fillna(self.df_numerical.mean())

        elif option == "mode":
            self.df_numerical = self.df_numerical.fillna(self.df_numerical.mode())

        elif option == "bfill":
            self.df_numerical = self.df_numerical.bfill()
            self.df_categorical = self.df_categorical.bfill()

        elif option == "ffill":
            self.df_numerical = self.df_numerical.ffill()
            self.df_categorical = self.df_categorical.ffill()

        elif option == "pairwise":
            self.df_numerical = self.df_numerical.dropna(how="any")
            self.df_categorical = self.df_categorical.dropna(how="any")

        elif option == "column":
            self.df_numerical = self.df_numerical.dropna(axis=1)
            self.df_categorical = self.df_categorical.dropna(axis=1)

        elif option == "row":
            self.df_numerical = self.df_numerical.dropna()
            self.df_categorical = self.df_categorical.dropna()

        else:
            print("Invalid option. Using default: median")
            self.df_numerical = self.df_numerical.fillna(self.df_numerical.median())

        self.df_final = pd.concat([self.df_numerical, self.df_categorical], axis=1)
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}", index=False)
        self.df_final.to_csv(
            f"{self.edit_csv_path}edit_remove_nulls_{self.csv_name}", index=False
        )
        return self.df_final.to_html()

    def encode(self, option: str) -> None:
        encoder = LabelEncoder()
        if option == "onehot":
            self.df_categorical = pd.get_dummies(self.df_categorical)
        elif option == "label":
            self.df_categorical = self.df_categorical.apply(encoder.fit_transform)
        else:
            print("Invalid option. Using default: label")
            self.df_categorical = self.df_categorical.apply(encoder.fit_transform)

        self.df_final = pd.concat([self.df_numerical, self.df_categorical], axis=1)
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}", index=False)
        self.df_final.to_csv(
            f"{self.edit_csv_path}edit_encode_{self.csv_name}", index=False
        )
        return self.df_final.to_html()

    def feature_scaling(self, option: str) -> None:
        if option == "standard":
            self.df_numerical = (
                self.df_numerical - self.df_numerical.mean()
            ) / self.df_numerical.std()
        elif option == "minmax":
            self.df_numerical = (self.df_numerical - self.df_numerical.min()) / (
                self.df_numerical.max() - self.df_numerical.min()
            )
        else:
            print("Invalid option. Using default: standard")
            self.df_numerical = (
                self.df_numerical - self.df_numerical.mean()
            ) / self.df_numerical.std()
        self.df_final = pd.concat([self.df_numerical, self.df_categorical], axis=1)
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}", index=False)
        self.df_final.to_csv(
            f"{self.edit_csv_path}edit_feature_scaling_{self.csv_name}", index=False
        )
        return self.df_final.to_html()
