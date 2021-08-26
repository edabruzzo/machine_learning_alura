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


sc = StandardScaler()
sc.fit(passageiros)
dado_escalado = sc.transform(passageiros)
x_escalado = dado_escalado[:, 0]  # Features - Características - Tempo
y_escalado = dado_escalado[:, 1]  # Alvo - Número de passageiros


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


# In[7]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)


plotar_eixos(x_treino, y_treino, 'treino')
plotar_eixos(x_teste, y_teste, 'teste')


# # Aula 2
#
# ## Regressão Linear

# In[8]:


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


# In[9]:


def plotar_resultados(x, y):
    dados = {'tempo': x, 'passageiros': y[:, 0]}
    resultados = pd.DataFrame(data=dados)
    resultados_inversos = sc.inverse_transform(resultados)
    x, y = resultados_inversos[:, 0], resultados_inversos[:, 1]
    plotar_eixos(x, y, 'predições')


# In[10]:


def testar_modelo(hyperparams, epocas_treino=5):
    dados = passageiros

    modelo = definir_modelo(hyperparams)
    modelo.fit(x_treino, y_treino, epochs=epocas_treino)
    y_predict = modelo.predict(x_treino)
    y_predict_teste = modelo.predict(x_teste)

    plotar_dados_completos()
    plotar_resultados(x_treino, y_predict)
    plotar_resultados(x_teste, y_predict_teste)


# In[11]:


hyperparams_1 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': True,
    'input_dim': 1
}]

testar_modelo(hyperparams=hyperparams_1)

# ## Regressão não-linear

# In[12]:


hyperparams_4 = [{

    'dimensao_saida': 8,
    'activation': 'sigmoid',
    'kernel_initializer': 'random_uniform',
    'use_bias': False,
    'input_dim': 1

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

testar_modelo(hyperparams_4, epocas_treino=500)


# # Aula 3

# ## Alterando a forma como passamos os dados
#
# Agora x e y vão valores diferentes. X vai conter o número de passageiros em um tempo anterior e y vai conter o número de passageiros em t+1, por exemplo.

# In[13]:


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


# In[14]:


n_passos = 1
xtreino_novo, ytreino_novo = separa_dados(y_treino, n_passos)
xteste_novo, yteste_novo = separa_dados(y_teste, n_passos)

# ## Agora vamos separar o teste

# ## Voltando para as redes neurais

# In[15]:


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

# In[16]:


regressor3.fit(xtreino_novo, ytreino_novo, epochs=100)

# In[17]:


y_predict_novo = regressor3.predict(xtreino_novo)


# In[18]:


def plotar_resultado_treino_dados_alterados(y, dados, label):
    sns.lineplot(x='tempo', y=y, data=dados, label=label)


# In[19]:


y_predict_teste_novo = regressor3.predict(xteste_novo)

# In[20]:


resultado = pd.DataFrame(y_predict_teste_novo)[0]


# In[32]:


def plotar_resultados_2():
    inicio = passageiros.shape[0] - yteste_novo.shape[0]
    fim = passageiros.shape[0]
    fim_2 = ytreino_novo.shape[0] + n_passos
    plotar_resultado_treino_dados_alterados(ytreino_novo, passageiros[n_passos:fim_2], 'treino')
    plotar_resultado_treino_dados_alterados(pd.DataFrame(y_predict_novo)[0], passageiros[n_passos:fim_2],
                                            'ajuste_treino')
    plotar_resultado_treino_dados_alterados(yteste_novo, passageiros[inicio:fim], 'teste')
    plotar_resultado_treino_dados_alterados(resultado.values, passageiros[inicio:fim], 'previsão')


plotar_resultados_2()

# ## Janelas

# In[22]:


n_passos = 4
xtreino_novo, ytreino_novo = separa_dados(y_treino, n_passos)
xteste_novo, yteste_novo = separa_dados(y_teste, n_passos)

# In[23]:


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

# In[24]:


regressor4.fit(xtreino_novo, ytreino_novo, epochs=300)

# In[25]:


y_predict_teste_novo = regressor4.predict(xteste_novo)

# In[26]:


resultado = pd.DataFrame(y_predict_teste_novo)[0]

# In[31]:


plotar_resultados_2()

# In[ ]:



