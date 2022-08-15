import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class preprocess:

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()

    def remove_columns_not_required(self, df,column_names):
        df.drop(columns=column_names, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def label_encode(self, original_df):
        integer_encoded = self.label_encoder.fit_transform(original_df)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return integer_encoded

    def one_hot_encode(self, original_df):
        integer_encoded = self.label_encode(original_df)
        onehot_encoded = self.onehot_encoder.fit_transform(integer_encoded)
        result = pd.DataFrame(onehot_encoded.toarray())
        return result
    
    def simple_impute(self, original_df: pd.DataFrame, column_name: list, strat="most_frequent") -> pd.DataFrame:
        temp_df = original_df[column_name]
        imputer = si(missing_values=np.nan, strategy=strat)
        imputer.fit(temp_df)
        original_df[column_name] = pd.DataFrame(imputer.transform(temp_df))
        return original_df

    def main(self, df):
        df = self.remove_columns_not_required(df,["Cabin","Name"])
        df[["em_1","em_2", "em_3"]] = self.encode(df["Embarked"])
        df[["Sex_1","Sex_2"]] = self.encode(df["Sex"])
        df["Ticket"] = self.label_encode(df["Ticket"])
        df.drop(columns=["Sex", "Embarked"], axis=1, inplace=True)
        return df

# obj = preprocess()
# test = pd.read_csv("/Users/prajualpillai/Desktop/prajual/Personal_git/boosting/test.csv")
# obj.main(test)
    
