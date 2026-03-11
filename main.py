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
    if not isinstance(texto, str) or texto == "":
        return ""
    print(f"Original: {texto[:50]}...")
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d+', '',texto)
    texto = re.sub(r'[^\w\s]', '',texto)
    tokens = nltk.word_tokenize(texto)
    print(f"Após Tokenizar: {tokens[:5]}")
    paradas = set(stopwords.words('portuguese'))
    tokens = [t for t in tokens if t not in paradas]
    print(f"Após Stopwords: {tokens[:5]}")
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    resultado = " ".join(tokens)
    print(f"Resultado Final: {resultado[:50]}")
    return resultado

df['preprocessed_news'] = df['preprocessed_news'].apply(limpeza_dataframe)

print(df.head())
