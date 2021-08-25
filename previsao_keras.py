#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install -r requirements.txt')

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

'''

cursos.alura.com.br/course/deep-learning-previsao-keras

github.com/alura-cursos/deeptime



'''

# In[3]:


mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22

# In[4]:


passageiros = pd.read_csv('Passageiros.csv')

# In[5]:


passageiros.head()


# In[6]:


def plotar_dados(dados):
    sns.lineplot(x='tempo', y='passageiros', data=dados, label='dado_completo')


plotar_dados(passageiros)

# In[7]:


sc = StandardScaler()
sc.fit(passageiros)
dado_escalado = sc.transform(passageiros)
dado_escalado

# In[8]:


x = dado_escalado[:, 0]
y = dado_escalado[:, 1]
x_treino = x[0:int(len(passageiros) * 0.8)]
y_treino = y[0:int(len(passageiros) * 0.8)]
x_teste = x[int(len(passageiros) * 0.8):len(passageiros)]
y_teste = y[int(len(passageiros) * 0.8):len(passageiros)]


# In[9]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)
    plt.ylabel('Passageiros')
    plt.xlabel('Data')


#plotar_eixos(x, y, 'dados_escalados')


# In[10]:


def definir_modelo(hyperparams=[], loss='mean_squared_error', optimizer='adam'):
    modelo = Sequential()

    for i in range(0, len(hyperparams)):
        modelo.add(Dense(hyperparams[i]['dimensao_saida'],
                         input_dim=1,
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

    plotar_dados(passageiros)
    plotar_resultados(x_treino, y_predict)
    plotar_resultados(x_teste, y_predict_teste)


# In[13]:


hyperparams_1 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': True,
}]

#testar_modelo(hyperparams_1)

# In[14]:


hyperparams_2 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': False,

}]

#testar_modelo(hyperparams_2)

# In[15]:


hyperparams_3 = [{

    'dimensao_saida': 8,
    'activation': 'linear',
    'kernel_initializer': 'random_uniform',
    'use_bias': False,

},

    {
        'dimensao_saida': 8,
        'activation': 'linear',
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

#testar_modelo(hyperparams_3, epocas_treino=100)

# In[16]:


hyperparams_4 = [{

    'dimensao_saida': 8,
    'activation': 'sigmoid',
    'kernel_initializer': 'random_uniform',
    'use_bias': True,

},

    {
        'dimensao_saida': 8,
        'activation': 'sigmoid',
        'kernel_initializer': 'random_uniform',
        'use_bias': True,

    },

    {
        'dimensao_saida': 1,
        'activation': 'linear',
        'kernel_initializer': 'random_uniform',
        'use_bias': True,

    }

]

#testar_modelo(hyperparams_4, epocas_treino=500)


# In[20]:


def altera_perspectiva(dados_mesmo_eixo_y, passos_atraso):
    X_novo, y_novo = [], []

    vetor = pd.DataFrame(dados_mesmo_eixo_y)[0]

    for i in (range(passos_atraso, vetor.shape[0])):
        X_novo.append((vetor.loc[i - passos_atraso]))
        y_novo.append(vetor.loc[i])

    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo


# In[39]:


def testar_modelo_modificado(hyperparams, epocas_treino=5, x_train=None, y_train=None, x_test=None, y_test=None):
    if (x_train is None):
        x_train = x_treino
        y_train = y_treino
        x_test = x_teste
        y_test = y_teste

    modelo = definir_modelo(hyperparams)
    modelo.fit(x_train, y_train, epochs=epocas_treino)
    y_predict = modelo.predict(x_train)
    y_predict_teste = modelo.predict(x_test)

    y_data_predict = pd.DataFrame(y_predict)[0]
    plotar_dados(passageiros)
    sns.lineplot(x='tempo', y=y_treino_novo, data=passageiros[1:115], label='treino')
    sns.lineplot(x='tempo', y=y_data_predict, data=passageiros[1:115], label='ajuste_treino')



# In[40]:


hyperparams_5 = [{

    'dimensao_saida': 8,
    'activation': 'linear',
    'kernel_initializer': 'ones',
    'use_bias': False,

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

X_treino_novo, y_treino_novo = altera_perspectiva(y_treino, 1)
X_teste_novo, y_teste_novo = altera_perspectiva(y_teste, 1)

'''
testar_modelo_modificado(hyperparams_5,
                         epocas_treino=100,
                         x_train=X_treino_novo,
                         y_train=y_treino_novo,
                         x_test=X_teste_novo,
                         y_test=y_teste_novo)
'''

plotar_eixos(X_treino_novo, y_treino_novo, 'dados_alterados_treino')
plotar_eixos(X_teste_novo, y_teste_novo, 'dados_alterados_teste')
