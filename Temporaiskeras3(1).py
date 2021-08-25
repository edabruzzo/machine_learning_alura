#!/usr/bin/env python
# coding: utf-8

# # Aula 1

# ## Carregando os dados

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22


# In[2]:


def carregar_dados():
    passageiros = pd.read_csv('Passageiros.csv')
    print(passageiros.head())
    return passageiros


passageiros = carregar_dados()


# In[3]:


def plotar_dados_completos():
    sns.lineplot(x='tempo', y='passageiros', data=passageiros, label='dado_completo')


# ## Escalando os dados

# In[4]:


def escalar_dados():
    sc = StandardScaler()
    sc.fit(passageiros)
    dado_escalado = sc.transform(passageiros)
    x = dado_escalado[:, 0]  # Features - Características - Tempo
    y = dado_escalado[:, 1]  # Alvo - Número de passageiros
    return x, y


x_escalado, y_escalado = escalar_dados()


# In[5]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)
    plt.ylabel('Passageiros')
    plt.xlabel('Data')


plotar_eixos(x_escalado, y_escalado, 'dado_escalado')


# ## Dividindo em treino e teste

# In[6]:


def dividir_treino_teste(tamanho_percentual_treino):
    tamanho_treino = int(len(passageiros) * tamanho_percentual_treino)
    tamanho_teste = int(len(passageiros) - tamanho_treino)
    xtreino = x_escalado[0:tamanho_treino]
    ytreino = y_escalado[0:tamanho_treino]
    xteste = x_escalado[tamanho_treino:len(passageiros)]
    yteste = y_escalado[tamanho_treino:len(passageiros)]
    return xtreino, ytreino, xteste, yteste


x_treino, y_treino, x_teste, y_teste = dividir_treino_teste(0.8)


# In[9]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)


plotar_eixos(x_treino, y_treino, 'treino')
plotar_eixos(x_teste, y_teste, 'teste')


# # Aula 2
#
# ## Regressão Linear

# In[10]:


def definir_modelo(hyperparams=[], input_dim=1, loss='mean_squared_error', optimizer='adam'):
    modelo = Sequential()

    if (hyperparams[0]['input_dim'] is not None):
        input_dim = hyperparams[0]['input_dim']

    for i in range(0, len(hyperparams)):

        modelo.add(Dense(hyperparams[i]['dimensao_saida'],
                         input_dim=input_dim,
                         activation=hyperparams[i]['activation'],
                         kernel_initializer=hyperparams[i]['kernel_initializer'],
                         use_bias=hyperparams[i]['use_bias']))

    modelo.compile(loss=loss,
                   metrics=['accuracy'],
                   optimizer=optimizer)

    modelo.summary()

    return modelo


# In[11]:


def plotar_resultados(x, y):
    dados = {'tempo': x, 'passageiros': y[:, 0]}
    resultados = pd.DataFrame(data=dados)
    resultados_inversos = sc.inverse_transform(resultados)
    x, y = resultados_inversos[:, 0], resultados_inversos[:, 1]
    plotar_eixos(x, y, 'predições')


# In[12]:


def testar_modelo(hyperparams, epocas_treino=5):
    dados = passageiros

    modelo = definir_modelo(hyperparams)
    modelo.fit(x_treino, y_treino, epochs=epocas_treino)
    y_predict = modelo.predict(x_treino)
    y_predict_teste = modelo.predict(x_teste)

    plotar_dados_completos()
    plotar_resultados(x_treino, y_predict)
    plotar_resultados(x_teste, y_predict_teste)


# In[13]:


hyperparams_1 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': True,
}]

# testar_modelo(hyperparams=hyperparams_1)


# ## Regressão não-linear

# In[14]:


hyperparams_4 = [{

    'dimensao_saida': 8,
    'activation': 'sigmoid',
    'kernel_initializer': 'random_uniform',
    'use_bias': False
},

    {
        'dimensao_saida': 8,
        'activation': 'sigmoid',
        'kernel_initializer': 'random_uniform',
        'use_bias': False

    },

    {
        'dimensao_saida': 1,
        'activation': 'linear',
        'kernel_initializer': 'random_uniform',
        'use_bias': False

    }

]


# testar_modelo(hyperparams_4, epocas_treino=500)


# # Aula 3

# ## Alterando a forma como passamos os dados
#
# Agora x e y vão valores diferentes. X vai conter o número de passageiros em um tempo anterior e y vai conter o número de passageiros em t+1, por exemplo.

# In[15]:


def separa_dados(dados_seriados, n_passos):
    """Entrada: vetor: número de passageiros
                 n_passos: número de passos no regressor
       Saída:
                X_novo: Array 2D
                y_novo: Array 1D - Nosso alvo
    """
    vetor = pd.DataFrame(dados_seriados)[0]
    X_novo, y_novo = [], []
    for i in range(n_passos, vetor.shape[0]):
        X_novo.append(list(vetor.loc[i - n_passos:i - 1]))
        y_novo.append(vetor.loc[i])
    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo


# In[16]:

n_passos = 1
xtreino_novo, ytreino_novo = separa_dados(y_treino, n_passos)
xteste_novo, yteste_novo = separa_dados(y_teste, n_passos)

# ## Agora vamos separar o teste

# ## Voltando para as redes neurais

# In[17]:


hyperparams_5 = [{

    'dimensao_saida': 8,
    'activation': 'linear',
    'kernel_initializer': 'ones',
    'use_bias': False,
    'input_dim': n_passos

},

    {
        'dimensao_saida': 64,
        'activation': 'sigmoid',
        'kernel_initializer': 'random_uniform',
        'use_bias': False,

    },

    {
        'dimensao_saida': 1,
        'activation': 'linear',
        'kernel_initializer': 'random_uniform',
        'use_bias': False,

    }

]

regressor3 = definir_modelo(hyperparams_5)

# In[18]:


regressor3.fit(xtreino_novo, ytreino_novo, epochs=100)

# In[19]:


y_predict_novo = regressor3.predict(xtreino_novo)


# In[21]:


def plotar_resultado_treino_dados_alterados(y, dados, label):
    sns.lineplot(x='tempo', y=y, data=dados, label=label)


# In[23]:


y_predict_teste_novo = regressor3.predict(xteste_novo)

# In[25]:


resultado = pd.DataFrame(y_predict_teste_novo)[0]

# In[26]:


plotar_resultado_treino_dados_alterados(ytreino_novo, passageiros[1:115], 'treino')
plotar_resultado_treino_dados_alterados(pd.DataFrame(y_predict_novo)[0], passageiros[1:115], 'ajuste_treino')
plotar_resultado_treino_dados_alterados(yteste_novo, passageiros[116:144], 'teste')
plotar_resultado_treino_dados_alterados(resultado.values, passageiros[116:144], 'previsão')

# ## Janelas

# In[28]:

n_passos=4
xtreino_novo, ytreino_novo = separa_dados(y_treino, n_passos)
xteste_novo, yteste_novo = separa_dados(y_teste, n_passos)

# In[30]:


hyperparams_6 = [{

    'dimensao_saida': 8,
    'activation': 'linear',
    'kernel_initializer': 'random_uniform',
    'use_bias': False,
    'input_dim': n_passos

},

    {
        'dimensao_saida': 64,
        'activation': 'sigmoid',
        'kernel_initializer': 'random_uniform',
        'use_bias': False,

    },

    {
        'dimensao_saida': 1,
        'activation': 'linear',
        'kernel_initializer': 'random_uniform',
        'use_bias': False,

    }

]

regressor4 = definir_modelo(hyperparams_6)

# In[32]:


regressor4.fit(xtreino_novo, ytreino_novo, epochs=300)

# In[33]:


y_predict_teste_novo = regressor4.predict(xteste_novo)

# In[34]:


resultado = pd.DataFrame(y_predict_teste_novo)[0]

# In[36]:


plotar_resultado_treino_dados_alterados(ytreino_novo, passageiros[4:115], 'treino')
plotar_resultado_treino_dados_alterados(pd.DataFrame(y_predict_novo)[0], passageiros[4:115], 'ajuste_treino')
plotar_resultado_treino_dados_alterados(yteste_novo, passageiros[119:144], 'teste')
plotar_resultado_treino_dados_alterados(resultado.values, passageiros[119:144], 'previsão')

# In[ ]: