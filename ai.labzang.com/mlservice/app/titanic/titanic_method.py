from pathlib import Path
import pandas as pd
from app.titanic.titanic_dataset import TitanicDataSet
from icecream import ic

class TitanicMethod(object): 

    def __init__(self):
        self.dataset = TitanicDataSet()

    def new_model(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    def create_train(self, df: DataFrame, label: str) -> pd.DataFrame:
        return df.drop(columns=[label])

    def create_label(self, df: DataFrame, label: str) -> pd.DataFrame:
        return df[[label]]

    def drop_feature(self, df: DataFrame, *feature: str) -> pd.DataFrame:
        return df.drop(columns=[x for x in feature])

    def check_null(self, df: DataFrame) -> int:
        return int(df.isnull().sum().sum())

