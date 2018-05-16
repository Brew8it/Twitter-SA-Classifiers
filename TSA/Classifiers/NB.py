from sklearn.externals import joblib
from TSA.Preproc.Preproc import Preproc
import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.model = joblib.load("TSA/TrainedModels/NB_imp_SemEval.pkl")
        self.pre_process_text = Preproc()

    def predict(self, text):

        self.pre_process_text.loadOwnDataFrame(pd.DataFrame(data=[text], columns=['text']))
        self.pre_process_text.clean_data()
        df = self.pre_process_text.get_twitter_df()

        try:
            return self.model.predict(df.text)


        except:
            print("Data is not foromated correctly. Try preprocessing it...")
