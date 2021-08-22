#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install -r requirements.txt')

# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# In[3]:


mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22

# In[4]:


passageiros = pd.read_csv('Passageiros.csv')

# In[5]:


passageiros.head()


# In[12]:


def plotar_dados(dados):
    sns.lineplot(x='tempo', y='passageiros', data=dados, label='dado_completo')


plotar_dados(passageiros)

# In[10]:


sc = StandardScaler()
sc.fit(passageiros)
dado_escalado = sc.transform(passageiros)
dado_escalado

# In[15]:


x = dado_escalado[:, 0]
y = dado_escalado[:, 1]


# In[25]:


def plotar_eixos(x, y, label):
    sns.lineplot(x=x, y=y, label=label)
    plt.ylabel('Passageiros')
    plt.xlabel('Data')


plotar_eixos(x, y, 'dados_escalados')

# In[26]:


tamanho_treino = int(len(passageiros) * 0.8)
tamanho_teste = int(len(passageiros) * 0.2)

x_treino = x[0:tamanho_treino]
y_treino = y[0:tamanho_treino]
x_teste = x[tamanho_treino:len(passageiros)]
y_teste = y[tamanho_treino:len(passageiros)]

plotar_eixos(x_treino, y_treino, 'treino')
plotar_eixos(x_teste, y_teste, 'teste')
