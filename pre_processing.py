import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class preprocess:

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()

    def remove_columns_not_required(self, df,column_names):
        df.drop(columns=column_names, axis=1, inplace=True)
        df.dropna(how="any", axis=0, inplace=True)
        return df
    
    def encode(self, df_3):
        integer_encoded = self.label_encoder.fit_transform(df_3)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded
    
    def main(self, df):
        df = self.remove_columns_not_required(df,["Cabin","Name"])
        df[["em_1","em_2", "em_3"]] = self.encode(df["Embarked"])
        df[["Sex_1","Sex_2"]] = self.encode(df["Sex"])
        integer_encoded = self.label_encoder.fit_transform(df["Ticket"])
        df["Ticket"] = integer_encoded.reshape(len(integer_encoded),1)
        df.drop(columns=["Sex", "Embarked"], axis=1)
        return df
        
    
