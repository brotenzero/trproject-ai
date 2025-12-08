from pathlib import Path
import pandas as pd
from app.titanic.titanic_dataset import TitanicDataSet
from icecream import ic

class TitanicMethod(object): 

    def __init__(self):
        self.dataset = TitanicDataSet()

    def new_model(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    def create_train(self) -> pd.DataFrame:
        return self.new_model().drop(columns=['Survived'])

    def create_label(self) -> pd.DataFrame:
        return self.new_model()[['Survived']]

    def drop_feature(self, *feature: str) -> pd.DataFrame:
        df_train = self.create_train()
        feature_list = list(feature)
        df_dropped = df_train.drop(columns=feature_list)
        return df_dropped

    def null_check(self) -> int:
        ic('🔍 데이터 결측치 확인')
        df_train = self.create_train()
        null_count = df_train.isnull().sum().sum()
        return int(null_count)

