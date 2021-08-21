#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install -r requirements.txt')
# !sudo apt-get install tcl-dev tk-dev python-tk python3-tk
# https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# http://scipy-lectures.org/advanced/image_processing/
from scipy import ndimage
import numpy as np

# In[3]:


dataset = keras.datasets.fashion_mnist

((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = dataset.load_data()

print(len(imagens_treino))
print(imagens_treino.shape)
print(imagens_teste.shape)
print(len(identificacoes_teste))

# In[4]:


plt.imshow(imagens_treino[0])
plt.title(identificacoes_treino[0])
plt.colorbar()

# In[5]:


'''
https://github.com/zalandoresearch/fashion-mnist

0 	T-shirt/top
1 	Trouser
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag
9 	Ankle boot

'''
labels = ['T-shirt/top',
          'Trouser',
          'Pullover',
          'Dress',
          'Coat',
          'Sandal',
          'Shirt',
          'Sneaker',
          'Bag',
          'Ankle boot']

# In[6]:


for indice in range(10):
    plt.subplot(2, 5, indice + 1)
    plt.imshow(imagens_treino[indice])
    plt.title(labels[identificacoes_treino[indice]])

# In[7]:


# Normalização para diminuir a perda
imagens_treino = imagens_treino / float(255)

modelo = keras.Sequential(
    # entrada
    [keras.layers.Flatten(input_shape=(imagens_treino.shape[1], imagens_treino.shape[2])),
     # processamento
     # 256 unidades --> múltiplo de 2
     keras.layers.Dense(256, activation=tf.nn.relu),
     # keras.layers.Dense(128, activation=tf.nn.relu),
     # keras.layers.Dense(64, activation=tf.nn.relu),
     keras.layers.Dropout(0.3),
     # saida
     keras.layers.Dense(len(labels), activation=tf.nn.softmax)

     ])


# In[53]:


def treinar_modelo():
    modelo.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    historico_treinamento = modelo.fit(imagens_treino,
                                       identificacoes_treino,
                                       epochs=3,
                                       validation_split=0.2)

    return historico_treinamento


# In[54]:


historico_1 = treinar_modelo()
historico_1.history


# In[55]:


def plotar(historico, metrica):
    if (metrica == 'Acurácia'):
        parametro_treino = 'acc'
        parametro_validacao = 'val_acc'

    if (metrica == 'Perda'):
        parametro_treino = 'loss'
        parametro_validacao = 'val_loss'

    plt.plot(historico.history[parametro_treino])
    plt.plot(historico.history[parametro_validacao])
    plt.legend(['treino', 'avaliacao'])
    plt.title(metrica + ' por épocas')
    plt.xlabel('épocas')
    plt.ylabel(metrica)


# In[56]:


plotar(historico_1, 'Perda')

# In[57]:


plotar(historico_1, 'Acurácia')

# In[58]:


testes = modelo.predict(imagens_teste)

# In[59]:


perda, acuracia = modelo.evaluate(imagens_teste, identificacoes_teste)

# In[60]:


modelo.save('modelo.h5')

# In[61]:


modelo_salvo = load_model('modelo.h5')
testes_modelo_salvo = modelo_salvo.predict(imagens_teste)
print('Resultado teste: ', np.argmax(testes_modelo_salvo[1]))
print('Número imagem teste: ', identificacoes_teste[1])

# In[62]:


sumario = modelo_salvo.summary()
sumario

# In[63]:


layer_1 = modelo.layers[1]
pesos_camada_dense = layer_1.get_weights()[0]

# In[64]:


vieses_camada_dense = layer_1.get_weights()[1]

# In[65]:


pesos_camada_dense_zerados = np.zeros(pesos_camada_dense.shape)
pesos_camada_dense_aleatorios = np.random.rand(pesos_camada_dense.shape[0],pesos_camada_dense.shape[1])
layer_1.set_weights([pesos_camada_dense_aleatorios, vieses_camada_dense])

# In[66]:


historico_2 = treinar_modelo()

# In[67]:


plotar(historico_2, 'Acurácia')

# In[68]:


plotar(historico_2, 'Perda')
