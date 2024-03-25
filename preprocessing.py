import pandas as pd


class Preprocessing:
    df = None

    def remove_nulls(self, option: str) -> None:
        if option == "median":
            self.df = self.df.fillna(self.df.median())
        elif option == "mean":
            self.df = self.df.fillna(self.df.mean())
        elif option == "mode":
            self.df = self.df.fillna(self.df.mode())
        else:
            print("Invalid option. Using default: median")
            self.df = self.df.fillna(self.df.median())
    
    def encode(self, option: str) -> None:
        if option == "onehot":
            self.df = pd.get_dummies(self.df)
        elif option == "label":
            self.df = self.df.apply(lambda x: pd.factorize(x)[0])
        else:
            print("Invalid option. Using default: onehot")
            self.df = pd.get_dummies(self.df)
    
    def feature_scaling(self, option: str) -> None:
        if option == "standard":
            self.df = (self.df - self.df.mean()) / self.df.std()
        elif option == "minmax":
            self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        else:
            print("Invalid option. Using default: standard")
            self.df = (self.df - self.df.mean()) / self.df.std()