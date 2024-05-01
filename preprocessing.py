import pandas as pd


class Preprocessing:
    
    def __init__(self, df: pd.DataFrame, csv_name: str, final_csv_path: str, edit_csv_path: str) -> None:
        self.df = df
        self.df_catagorical = self.df.select_dtypes("object")
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
        else:
            print("Invalid option. Using default: median")
            self.df_numerical = self.df_numerical.fillna(self.df_numerical.median())
        self.df_final = pd.concat([self.df_numerical, self.df_catagorical], axis=1)  
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}")
        self.df_final.to_csv(f"{self.edit_csv_path}edit_remove_nulls_{self.csv_name}")
        return self.df_final.to_html()
    
    def encode(self, option: str) -> None:
        if option == "onehot":
            self.df_catagorical = pd.get_dummies(self.df_catagorical)
        elif option == "label":
            self.df_catagorical = self.df_catagorical.apply(lambda x: pd.factorize(x)[0])
        else:
            print("Invalid option. Using default: label")
            self.df_catagorical = self.df_catagorical.apply(lambda x: pd.factorize(x)[0])
        
        self.df_final = pd.concat([self.df_numerical, self.df_catagorical], axis=1)  
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}")
        self.df_final.to_csv(f"{self.edit_csv_path}edit_encode_{self.csv_name}")
        return self.df_final.to_html()
    
    def feature_scaling(self, option: str) -> None:
        if option == "standard":
            self.df = (self.df - self.df.mean()) / self.df.std()
        elif option == "minmax":
            self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        else:
            print("Invalid option. Using default: standard")
            self.df = (self.df - self.df.mean()) / self.df.std()
        self.df_final = self.df
        self.df_final.to_csv(f"{self.final_csv_path}final_{self.csv_name}")
        self.df_final.to_csv(f"{self.edit_csv_path}edit_feature_scaling_{self.csv_name}")
        return self.df_final.to_html()