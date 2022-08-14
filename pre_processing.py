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
        df.reset_index(drop=True, inplace=True)
        return df
    
    def encode(self, df_3):
        integer_encoded = self.label_encoder.fit_transform(df_3)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.fit_transform(integer_encoded)
        result = pd.DataFrame(onehot_encoded.toarray())
        return result

    def main(self, df):
        df = self.remove_columns_not_required(df,["Cabin","Name"])
        embarked_encoded = self.encode(df["Embarked"])
        df[["em_1","em_2", "em_3"]] = embarked_encoded
        df[["Sex_1","Sex_2"]] = self.encode(df["Sex"])
        integer_encoded = self.label_encoder.fit_transform(df["Ticket"])
        df["Ticket"] = integer_encoded.reshape(len(integer_encoded),1)
        df.drop(columns=["Sex", "Embarked"], axis=1, inplace=True)
        return df

# obj = preprocess()
# test = pd.read_csv("/Users/prajualpillai/Desktop/prajual/Personal_git/boosting/test.csv")
# obj.main(test)
    
