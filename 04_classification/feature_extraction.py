from typing import Tuple

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Featurizer():
    def __init__(self, drop_cols):
        self.drop_cols = drop_cols

        self.onehot = preprocessing.OneHotEncoder(drop="first")
        self.le = preprocessing.LabelEncoder()

    def get_train_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df.drop(self.drop_cols, axis=1, inplace=True)
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_df["Sex"] = self.le.fit_transform(train_df["Sex"])
        val_df["Sex"] = self.le.transform(val_df["Sex"])

        onehot_vals = self.onehot.fit_transform(train_df["Pclass"].values[:,None]).todense()
        self.class_names = self.onehot.categories_[0][1:]
        pclass = pd.DataFrame(onehot_vals, columns=self.class_names)
        train_df.drop("Pclass", axis=1, inplace=True)
        train_df = pd.concat([train_df, pclass], axis=1)

        onehot_vals = self.onehot.transform(val_df["Pclass"].values[:, None]).todense()
        pclass = pd.DataFrame(onehot_vals, columns=self.class_names)
        val_df.drop("Pclass", axis=1, inplace=True)
        val_df = pd.concat([val_df, pclass], axis=1)

        return train_df, val_df

    def get_test_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(self.drop_cols, axis=1, inplace=True)

        df["Sex"] = self.le.transform(df["Sex"])

        onehot_vals = self.onehot.transform(df["Pclass"].values[:, None]).todense()
        pclass = pd.DataFrame(onehot_vals, columns=self.class_names)
        df.drop("Pclass", axis=1, inplace=True)
        df = pd.concat([df, pclass], axis=1)

        return df