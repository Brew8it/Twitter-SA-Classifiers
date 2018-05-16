# coding=utf-8
import re
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import unicodedata

stemmer = SnowballStemmer('english')
stop = stopwords.words('english')
stop.remove('not')

# for expanding contractions

cont_dict = {'can\'t': 'cannot', 'won\'t': 'will not', 'n\'t': ' not'}
cont_re = re.compile('(%s)' % '|'.join(cont_dict.keys()))


class Preproc(object):
    def __init__(self):
        self.df = pd.DataFrame
        self.filename = ""

    def loadOwnDataFrame(self, dataframe):
        self.df = dataframe.dropna()

    def expand_cont(self, s, cont_dict=cont_dict):
        def replace(match):
            return cont_dict[match.group(0)]

        return cont_re.sub(replace, s)

    def clean_data(self, html_strpping=True, accented_char_removal=True, to_lower=True, remove_links=True,
                   remove_mentions=True,
                   remove_hashtag=True, remove_extra_whitespace=True, tokenize=True, stemming=True,
                   remake_document=True,
                   expand_contractions=True):
        if html_strpping:
            self.remove_html_encode()
        if to_lower:
            self.to_lower()
        if expand_contractions:
            self.expand_contractions()
        if accented_char_removal:
            self.remove_accented_chars()
        if remove_links:
            self.remove_links()
        if remove_mentions:
            self.remove_twitter_mention()
        if remove_hashtag:
            self.remove_hashtag()
        if remove_extra_whitespace:
            self.remove_extra_whitepsace()
        if tokenize:
            self.tokenize()
        if stemming:
            self.word_stemming()
        if stopwords:
            self.remove_stopwords()
        if remake_document:
            self.remake_texts()


    def get_twitter_df(self):
        return self.df

    def remove_html_encode(self):
        self.df["text"] = self.df["text"].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_accented_chars(self):
        self.df["text"] = self.df["text"].apply(
            lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    def to_lower(self):
        self.df["text"] = self.df["text"].str.lower()

    def remove_links(self):
        self.df.loc[:, "text"] = self.df.loc[:, "text"].replace(r'https?://[A-Za-z0-9./]+', '', regex=True)

    def remove_twitter_mention(self):
        self.df.loc[:, "text"] = self.df.loc[:, "text"].replace(r'@[A-Za-z0-9]+', '', regex=True)

    def remove_hashtag(self):
        self.df.loc[:, "text"] = self.df.loc[:, "text"].replace('[^a-zA-Z]', ' ', regex=True)

    def remove_extra_whitepsace(self):
        self.df.loc[:, "text"] = self.df.loc[:, "text"].replace(' +', ' ', regex=True)

    def expand_contractions(self):
        self.df["text"] = self.df["text"].apply(lambda x: self.expand_cont(x))

    def tokenize(self):
        self.df["text"] = self.df["text"].apply(word_tokenize)

    def word_stemming(self):
        self.df["text"] = self.df["text"].apply(lambda x: [stemmer.stem(y) for y in x])

    def remove_stopwords(self):
        self.df["text"] = self.df["text"].apply(lambda x: [item for item in x if item not in stop])

    def remake_texts(self):
        self.df["text"] = self.df["text"].apply(lambda x: ' '.join(x))