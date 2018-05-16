import pandas as pd
import numpy as np
from keras.models import model_from_json
from TSA.Preproc.Preproc import Preproc
from TSA.CNN.data_handler import TextHandlerCNN


def load_cnn():
    json_file = open("TSA/TrainedModels/CNN_base_SemEval.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights("TSA/TrainedModels/CNN_base_SemEval_w.h5")
    return model

def label_cnn_pred(pred_tuple):
    labels = [1, 0]
    arglist = []
    # Swap from [0.423, 0.677] -> lable = 0
    for p in pred_tuple:
        arglist.append(labels[np.argmax(p)])
    return arglist


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.model = load_cnn()
        self.pre_process_text = Preproc()


    def predict(self, text):
        self.pre_process_text.loadOwnDataFrame(pd.DataFrame(data=[text], columns=['text']))
        self.pre_process_text.clean_data()
        df = self.pre_process_text.get_twitter_df()
        handled_text = TextHandlerCNN().pred_load_data(df)

        try:
            return label_cnn_pred(self.model.predict(handled_text))
        except:
            print("Data is not foromated correctly. Try preprocessing it...")