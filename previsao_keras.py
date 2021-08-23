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


# In[9]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)
    plt.ylabel('Passageiros')
    plt.xlabel('Data')


plotar_eixos(x, y, 'dados_escalados')

# In[10]:


tamanho_treino = int(len(passageiros) * 0.8)
tamanho_teste = int(len(passageiros) * 0.2)

x_treino = x[0:tamanho_treino]
y_treino = y[0:tamanho_treino]
x_teste = x[tamanho_treino:len(passageiros)]
y_teste = y[tamanho_treino:len(passageiros)]

plotar_eixos(x_treino, y_treino, 'treino')
plotar_eixos(x_teste, y_teste, 'teste')


# In[21]:


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


# In[23]:


def plotar_resultados(x, y):

    dados = {'tempo': x, 'passageiros': y[:, 0]}
    resultados = pd.DataFrame(data=dados)
    resultados_inversos = sc.inverse_transform(resultados)
    x, y = resultados_inversos[:, 0], resultados_inversos[:, 1]
    plotar_eixos(x, y, 'predições')


# In[38]:


def testar_modelo(hyperparams, epocas_treino=5):

    modelo = definir_modelo(hyperparams)
    modelo.fit(x_treino, y_treino, epochs=epocas_treino)
    y_predict = modelo.predict(x_treino)
    y_predict_teste = modelo.predict(x_teste)
    plotar_dados(passageiros)
    plotar_resultados(x_treino, y_predict)
    plotar_resultados(x_teste, y_predict_teste)


# In[39]:


hyperparams_1 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': True,
}]

testar_modelo(hyperparams_1)

# In[40]:


hyperparams_2 = [{

    'dimensao_saida': 1,
    'activation': 'linear',
    'kernel_initializer': 'Ones',
    'use_bias': False,

}]

testar_modelo(hyperparams_2)

# In[42]:


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

# In[ ]:



