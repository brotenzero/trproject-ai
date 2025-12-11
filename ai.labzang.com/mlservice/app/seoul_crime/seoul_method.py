import pandas as pd
import numpy as np
from pandas import DataFrame
import logging
from app.seoul_crime.seoul_data import SeoulData

logger = logging.getLogger(__name__)

class SeoulMethod(object): 

    def __init__(self):
        self.dataset = SeoulData()

    def csv_to_df(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    def xlsx_to_df(self, fname: str) -> pd.DataFrame:
        return pd.read_excel(fname)

        