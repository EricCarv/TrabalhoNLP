import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import nltk
from PIL import Image
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
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
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d+', '',texto)
    texto = re.sub(r'[^\w\s]', '',texto)
    tokens = nltk.word_tokenize(texto)
    paradas = set(stopwords.words('portuguese'))
    tokens = [t for t in tokens if t not in paradas]
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    resultado = " ".join(tokens)
    return resultado
print("iniciou limpeza")
df['preprocessed_news'] = df['preprocessed_news'].apply(limpeza_dataframe)
print("terminou limpeza")
print(df.head())
print("iniciou regressão / TFIDF\n")
menor_tipo=df['label'].value_counts().min()
df_falso = df[df['label']== 'fake'].sample(n=menor_tipo, random_state=50)
df_real = df[df['label']== 'true'].sample(n=menor_tipo, random_state=50)
df_balanceado = pd.concat([df_falso,df_real])

tfidf = TfidfVectorizer(ngram_range=(1, 3))
x = tfidf.fit_transform(df_balanceado['preprocessed_news'])
y = df_balanceado['label']

X_treino, X_teste, Y_treino, Y_teste, texto_treino, texto_teste = train_test_split(x , y, df['preprocessed_news'], test_size=0.25, random_state=50)

regressao = LogisticRegression(solver='lbfgs', max_iter=2000)
regressao.fit(X_treino, Y_treino)

prob_pred = regressao.predict_proba(X_teste)
label_pred = regressao.predict(X_teste)
precisao = accuracy_score (Y_teste, label_pred)

print(f"Resultado da regressão")
print(f"Precisão geral do modelo : {precisao:.4f}")
print(f"Exemplos Probabilidade (Fake , Real):\n{prob_pred[:5]}")

df_resultados = pd.DataFrame({
    'texto': texto_teste,
    'previsao': label_pred
})

textos_verdade = " ".join(df_resultados[df_resultados['previsao'] == 'true']['texto'])
textos_fake = " ".join(df_resultados[df_resultados['previsao'] == 'fake']['texto'])

mascara_positiva = np.array(Image.open('thumbsup-svgrepo-com1.png').convert("L"))
mascara_negativa = np.array(Image.open('dull-mad-angry-emoji-emoticon-svgrepo-com1.png').convert("L"))


plt.imshow(mascara_positiva, cmap='gray')
plt.title("Máscara usada na WordCloud")
plt.show()

print(f"Shape da máscara: {mascara_positiva.shape}")
print(f"Pixels únicos na máscara: {np.unique(mascara_positiva)}")

wordcloud_positiva = WordCloud(
    background_color='black',
    width=1024,
    height=1024,
    mask=mascara_positiva,
    contour_width=3,
    contour_color='lightgreen',
    colormap='winter',
    max_words=150,
).generate(textos_verdade)

wordcloud_negativa = WordCloud(
    background_color='black',
    width=1024,
    height=1024,
    mask=mascara_negativa,
    contour_width=3,
    contour_color='firebrick',
    colormap='autumn',
    max_words=150,
).generate(textos_fake)

plt.figure(figsize= (15,10))
plt.title("teste titulo", fontsize=24, pad=20)
plt.imshow(wordcloud_positiva, interpolation= 'bilinear')
plt.axis('off')
plt.savefig("wordcloud_positiva.png")
plt.show()

plt.figure(figsize= (15,10))
plt.imshow(wordcloud_negativa, interpolation= 'bilinear')
plt.axis('off')
plt.savefig("wordcloud_negativa.png")
plt.show()
