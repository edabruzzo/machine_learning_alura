#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# Referência: https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html
from scipy import sparse
import time
# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
import os
from os import path
from wordcloud import WordCloud
import nltk
from nltk import tokenize
from nltk.corpus import stopwords

nltk.download('all')

# In[2]:


start_time = time.time()

resenhas_imdb = pd.read_csv('../../dados_imdb/imdb-reviews-pt-br.csv')

'''
# PARA MOSTRAR TODOS
#https://stackoverflow.com/questions/62207066/pandas-does-not-show-the-complete-csv-file
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    display(resenhas_imdb)
'''

resenhas_imdb.head()

# In[3]:


resenhas_imdb['classificacao'] = resenhas_imdb['sentiment'].replace(['neg', 'pos'], [0, 1])


# In[21]:


def treinar_classificador(com_tratamento=True):
    vetorizador = CountVectorizer(lowercase=False, max_features=400)

    tamanho_texto_integral = len(resenhas_imdb['text_pt'])

    if com_tratamento == True:

        palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
        from string import punctuation
        for pontuacao in punctuation:
            palavras_irrelevantes.append(pontuacao)

    else:
        palavras_irrelevantes = []

    frase_processada = list()

    # tokenizador = tokenize.WhitespaceTokenizer()
    tokenizador = tokenize.WordPunctTokenizer()

    # Retirada de acentos
    import unidecode

    for opiniao in resenhas_imdb.text_pt:
        palavras_texto = tokenizador.tokenize(opiniao)

        if com_tratamento == True:
            nova_frase = [unidecode.unidecode(palavra) for palavra in palavras_texto if
                          palavra not in palavras_irrelevantes]
        else:
            nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]

        frase_processada.append(' '.join(nova_frase))

    if com_tratamento == True:
        resenhas_imdb["texto_pt_tratado"] = frase_processada
        texto = 'texto_pt_tratado'

    else:
        texto = 'text_pt'

    bag_of_words = vetorizador.fit_transform(resenhas_imdb[texto])

    '''
    matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
                                                      columns=vetorizador.get_feature_names())
    '''

    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                                  resenhas_imdb.classificacao,
                                                                  random_state=42)

    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
    regressao_logistica = LogisticRegression(solver='lbfgs', max_iter=200)
    regressao_logistica.fit(treino, classe_treino)
    acuracia = regressao_logistica.score(teste, classe_teste)

    return acuracia


# ## Testando acurácia sem tratamento do texto

# In[22]:


# https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
'''
Testando acurácia sem tratamento do texto

'''
print('Acurácia %.2f%%' % (treinar_classificador(com_tratamento=False) * 100))

# In[24]:


resenhas_imdb["text_pt"]

# ## Testando acurácia com tratamento do texto

# In[23]:


# https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
print('Acurácia %.2f%%' % (treinar_classificador() * 100))

resenhas_imdb["texto_pt_tratado"]

# In[6]:


tempo_execucao = time.time() - start_time
print('-----Tempo de execução: %s segundos' % tempo_execucao)
