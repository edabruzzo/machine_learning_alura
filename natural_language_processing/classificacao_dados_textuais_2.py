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


# In[4]:


def treinar_classificador(com_tratamento=True,
                          utiliza_stemmer=False,
                          utilizar_TFIDF=True,
                          utilizar_ngramas=True,
                          maximo_palavras=50):
    tamanho_texto_integral = len(resenhas_imdb['text_pt'])

    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
    regressao_logistica = LogisticRegression(solver='lbfgs', max_iter=200)

    if com_tratamento:

        palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
        from string import punctuation
        for pontuacao in punctuation:
            palavras_irrelevantes.append(pontuacao)

    else:
        palavras_irrelevantes = []

    frase_processada = list()

    # tokenizador = tokenize.WhitespaceTokenizer()
    tokenizador = tokenize.WordPunctTokenizer()
    stemmer = nltk.stem.RSLPStemmer()

    if utilizar_TFIDF:

        from sklearn.feature_extraction.text import TfidfVectorizer
        if utilizar_ngramas:
            tf_idf = TfidfVectorizer(lowercase=False, max_features=maximo_palavras, ngram_range=(1, 2))
        else:
            tf_idf = TfidfVectorizer(lowercase=False, max_features=maximo_palavras)


    else:
        vetorizador = CountVectorizer(lowercase=False, max_features=maximo_palavras)

    # Retirada de acentos
    import unidecode

    jah_houve_tratamento_previo = 'texto_pt_tratado' in resenhas_imdb.columns

    if ((com_tratamento and not jah_houve_tratamento_previo) or com_tratamento == False):

        if not jah_houve_tratamento_previo:
            print('Necessário inicializar o tratamento no primeiro teste')

        for opiniao in resenhas_imdb.text_pt:

            palavras_texto = tokenizador.tokenize(opiniao)

            if com_tratamento:

                if utiliza_stemmer:
                    nova_frase = [stemmer.stem(unidecode.unidecode(palavra).lower()) for palavra in palavras_texto if
                                  palavra not in palavras_irrelevantes]
                else:
                    nova_frase = [unidecode.unidecode(palavra).lower() for palavra in palavras_texto if
                                  palavra not in palavras_irrelevantes]
            else:
                nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]

            frase_processada.append(' '.join(nova_frase))

    if com_tratamento:
        if not jah_houve_tratamento_previo:
            resenhas_imdb["texto_pt_tratado"] = frase_processada
        texto = 'texto_pt_tratado'

    else:
        texto = 'text_pt'

    if utilizar_TFIDF:
        matriz_treinamento = tf_idf.fit_transform(resenhas_imdb[texto])
    else:
        matriz_treinamento = vetorizador.fit_transform(resenhas_imdb[texto])

    '''
    matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
                                                      columns=vetorizador.get_feature_names())
    '''

    treino, teste, classe_treino, classe_teste = train_test_split(matriz_treinamento,
                                                                  resenhas_imdb.classificacao,
                                                                  random_state=42)

    regressao_logistica.fit(treino, classe_treino)
    acuracia = regressao_logistica.score(teste, classe_teste)

    if utilizar_TFIDF:
        pesos = pd.DataFrame(regressao_logistica.coef_[0].T,
                             index=tf_idf.get_feature_names())
    else:
        pesos = pd.DataFrame(regressao_logistica.coef_[0].T,
                             index=vetorizador.get_feature_names())

    print(pesos.nlargest(10, 0))
    print(pesos.nsmallest(10, 0))

    return acuracia


# ## Testando acurácia em diferentes configurações

# In[5]:


def testar_configuracoes_treinamento(max_palavras=50):
    # https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
    print('Acurácia sem tratamento do texto e sem tf-idf: %.2f%%'
          % (treinar_classificador(com_tratamento=False,
                                   utilizar_TFIDF=False,
                                   maximo_palavras=max_palavras
                                   ) * 100))

    # https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
    print('Acurácia sem tratamento do texto e com tf_idf: %.2f%%'
          % (treinar_classificador(com_tratamento=False,
                                   utilizar_TFIDF=True,
                                   maximo_palavras=max_palavras
                                   ) * 100))
    # https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
    print('Acurácia com tratamento do texto e utilizaçao de TF-IDF SEM NGRAM(1,2): %.2f%%'
          % (treinar_classificador(com_tratamento=True,
                                   utiliza_stemmer=False,
                                   utilizar_TFIDF=True,
                                   utilizar_ngramas=False,
                                   maximo_palavras=max_palavras
                                   ) * 100))

    # https://www.ic.unicamp.br/~mc102/mc102-1s2019/labs/format.html
    # treinar_classificador(com_tratamento=True, utiliza_stemmer=False, utilizar_TFIDF=True, utilizar_ngramas=True)
    print('Acurácia com tratamento do texto e utilizaçao de TF-IDF COM NGRAM(1,2): %.2f%%'
          % (treinar_classificador(
        maximo_palavras=max_palavras

    ) * 100))


inicio_teste_1 = time.time()

testar_configuracoes_treinamento()

tempo_execucao_teste_1 = time.time() - inicio_teste_1
print('-----Tempo de execução do teste para 50 palavras no treinamento: %s segundos' % tempo_execucao_teste_1)

# In[6]:


inicio_teste_2 = time.time()

testar_configuracoes_treinamento(max_palavras=500)

tempo_execucao_teste_2 = time.time() - inicio_teste_2
print('-----Tempo de execução do teste para 500 palavras no treinamento: %s segundos' % tempo_execucao_teste_2)

# In[8]:


inicio_teste_3 = time.time()

testar_configuracoes_treinamento(max_palavras=1000)

tempo_execucao_teste_3 = time.time() - inicio_teste_3
print('-----Tempo de execução do teste para 1.000 palavras no treinamento: %s segundos' % tempo_execucao_teste_2)

# In[7]:


tempo_execucao = time.time() - start_time
print('-----Tempo de execução: %s segundos' % tempo_execucao)
