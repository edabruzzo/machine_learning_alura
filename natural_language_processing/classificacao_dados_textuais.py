
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Referência: https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html
from scipy import sparse


import time

start_time = time.time()


resenhas_imdb = pd.read_csv('dados_imdb/imdb-reviews-pt-br.csv')


resenhas_imdb['classificacao'] = resenhas_imdb['sentiment'].replace(['neg', 'pos'], [0,1])

print(resenhas_imdb['sentiment'].value_counts())
print(resenhas_imdb['classificacao'].value_counts())

vetorizador = CountVectorizer(lowercase=False, max_features=50)

bag_of_words = vetorizador.fit_transform(resenhas_imdb['text_pt'])

print(bag_of_words.shape)


matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
                                                   columns=vetorizador.get_feature_names())



treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              resenhas_imdb.classificacao,
                                                              random_state = 42)


regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)
print(acuracia)

tempo_execucao = time.time() - start_time
print('-----Tempo de execução: %s segundos' % tempo_execucao)