#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -r requirements.txt')

# In[17]:


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

# In[7]:


mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22

# In[8]:


passageiros = pd.read_csv('Passageiros.csv')

# In[9]:


passageiros.head()


# In[10]:


def plotar_dados(dados):
    sns.lineplot(x='tempo', y='passageiros', data=dados, label='dado_completo')


plotar_dados(passageiros)

# In[11]:


sc = StandardScaler()
sc.fit(passageiros)
dado_escalado = sc.transform(passageiros)
dado_escalado

# In[12]:


x = dado_escalado[:, 0]
y = dado_escalado[:, 1]


# In[13]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)
    plt.ylabel('Passageiros')
    plt.xlabel('Data')


plotar_eixos(x, y, 'dados_escalados')

# In[14]:


tamanho_treino = int(len(passageiros) * 0.8)
tamanho_teste = int(len(passageiros) * 0.2)

x_treino = x[0:tamanho_treino]
y_treino = y[0:tamanho_treino]
x_teste = x[tamanho_treino:len(passageiros)]
y_teste = y[tamanho_treino:len(passageiros)]

plotar_eixos(x_treino, y_treino, 'treino')
plotar_eixos(x_teste, y_teste, 'teste')


# In[28]:


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


# In[29]:


def plotar_resultados(x, y):
    dados = {'tempo': x, 'passageiros': y[:, 0]}
    resultados = pd.DataFrame(data=dados)
    resultados_inversos = sc.inverse_transform(resultados)
    x, y = resultados_inversos[:, 0], resultados_inversos[:, 1]
    plotar_eixos(x, y, 'predições')


# In[30]:


def testar_modelo(hyperparams, epocas_treino=5):
    modelo = definir_modelo(hyperparams)
    modelo.fit(x_treino, y_treino, epochs=epocas_treino)
    y_predict = modelo.predict(x_treino)
    y_predict_teste = modelo.predict(x_teste)
    plotar_dados(passageiros)
    plotar_resultados(x_treino, y_predict)
    plotar_resultados(x_teste, y_predict_teste)


# In[14]:


hyperparams_1 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': True,
}]

testar_modelo(hyperparams_1)

# In[15]:


hyperparams_2 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': False,

}]

testar_modelo(hyperparams_2)

# In[16]:


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

testar_modelo(hyperparams_3, epocas_treino=100)

# In[17]:


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

testar_modelo(hyperparams_4, epocas_treino=500)


# In[20]:


def altera_perspectiva(dados, passos_atraso):
    X_novo, y_novo = [], []

    vetor = pd.DataFrame(dados)[0]

    for i in (range(passos_atraso, vetor.shape[0])):
        X_novo.append(list(vetor.loc[i - passos_atraso:i - 1]))
        y_novo.append(vetor.loc[i])

    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo


# In[25]:


X_treino_novo, y_treino_novo = altera_perspectiva(y_treino, 1)
print(X_treino_novo[0:5])
print(y_treino_novo[0:5])

# In[26]:


X_teste_novo, y_teste_novo = altera_perspectiva(y_teste, 1)
print(X_teste_novo[0:5])
print(y_teste_novo[0:5])

# In[31]:


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

testar_modelo(hyperparams_5, epocas_treino=100)

# In[36]:


hyperparams_5 = [{

    'dimensao_saida': 8,
    'activation': 'linear',
    'kernel_initializer': 'ones',
    'use_bias': True,

},

    {
        'dimensao_saida': 64,
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

testar_modelo(hyperparams_5, epocas_treino=500)

# In[ ]:



