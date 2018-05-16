import numpy as np
from sklearn.externals import joblib


class TextHandlerCNN:
    def get_text_as_numbers_pred(self, text, vocabulary):
        a = []

        for sentence in text:
            numbers = []
            for word in sentence:
                if word in vocabulary:
                    numbers.append(vocabulary[word])
            while len(numbers) < 64:
                numbers.append(vocabulary["<PAD/>"])

            a.append(numbers)

        return np.array(a)

    def pad_text(self, text, max_length, padding="<PAD/>"):
        return text + [padding] * (max_length - len(text))

    def pred_load_data(self, df, vocab_name="vocabulary_SE"):
        # Convert text to list of words and apply padding
        df["text"] = df["text"].apply(lambda x: x.split(" "))
        # set max length to a fixed size..
        max_length = 64
        df["text"] = df["text"].apply(lambda x: self.pad_text(x, max_length))
        vocabulary = joblib.load("TSA/CNN/" + vocab_name + ".pkl")

        x = self.get_text_as_numbers_pred(df.text, vocabulary)

        return x
