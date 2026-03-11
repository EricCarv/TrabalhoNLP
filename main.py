import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, ImageColorGenerator
from nltk import ngrams, RSLPStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('pre-processed.csv')
#nltk.download("all")
#print(df.head())

def limpeza_dataframe(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d+', '',texto)
    texto = re.sub(r'[\w\s]', '',texto)
    tokens = nltk.word_tokenize(texto)
    paradas = set(stopwords('portuguese'))
    tokens = [t for t in texto if t not in paradas]
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
