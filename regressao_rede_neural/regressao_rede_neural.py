#!/usr/bin/env python
# coding: utf-8

# # A base de dados

# ### Carregando o dataset

# In[1]:


# from google.colab import files
# files.upload()


# In[2]:


import pandas as pd

dados = pd.read_csv('./data/Bicicletas.csv')

# ### Conhecendo a base de dados

# In[3]:


dados.head()

# In[4]:


dados.shape

# In[5]:


import matplotlib.pyplot as plt


# In[6]:


def plotar_dados(x, y, label_eixoX, indices=None):
    if indices is not None:
        plt.xticks(indices, fontsize=14)

    plt.rcParams.update({'font.size': 14})
    plt.scatter(x, y)
    plt.xlabel(label_eixoX)
    plt.ylabel('bicicletas_alugadas')


plotar_dados(dados['temperatura'], dados['bicicletas_alugadas'], label_eixoX='temperatura')

# In[7]:


plotar_dados(dados['clima'], dados['bicicletas_alugadas'], label_eixoX='clima', indices=[1, 2, 3])

# ### Normalizando a base de dados

# In[8]:


import numpy as np

# In[9]:


y = dados['bicicletas_alugadas'].values

# In[10]:


X = dados[['clima', 'temperatura']].values
print(X)

# In[11]:


X = X / np.amax(X, axis=0)
print(X)

# In[12]:


ymax = np.amax(y)
y = y / ymax
print(y[0:10])

# ### Funções de ativação

# In[13]:


sigmoid = lambda valor: 1 / (1 + np.exp(-valor))

relu = lambda valor: np.maximum(0, valor)

# ### Criando a estrutura da rede

# In[14]:


arquiteturas = []

arquitetura_1 = [
    {"dim_entrada": 2, "dim_saida": 50, "ativacao": "relu"},
    {"dim_entrada": 50, "dim_saida": 1, "ativacao": "sigmoid"}
]

arquitetura_2 = [
    {"dim_entrada": 2, "dim_saida": 4, "ativacao": "sigmoid"},
    {"dim_entrada": 4, "dim_saida": 1, "ativacao": "sigmoid"}
]

arquiteturas.append(arquitetura_1)
arquiteturas.append(arquitetura_2)


# ### Pesos e viés

# In[15]:


def inicia_camadas(arquitetura, seed=99):
    # inicia os valores aleatórios
    np.random.seed(seed)
    # numero de camadas da rede neural
    numero_de_camadas = len(arquitetura)
    # inicia armazenamento de parametros
    valores_parametros = {}

    # itera nas camadas da rede
    for indice, camada in enumerate(arquitetura):
        indice_camada = indice + 1

        # extrai o numero de nodos nas camadas
        tamanho_camada_entrada = camada["dim_entrada"]
        tamanho_camada_saida = camada["dim_saida"]

        # inicia os valores na matriz de pesos P
        # e o vetor de viés ou bias b
        valores_parametros['P' + str(indice_camada)] = np.random.randn(
            tamanho_camada_saida, tamanho_camada_entrada) * 0.1
        valores_parametros['b' + str(indice_camada)] = np.random.randn(
            tamanho_camada_saida, 1) * 0.1

    return valores_parametros


# ### Propagação da rede

# In[16]:


def propaga_uma_camada(Ativado_anterior, Pesos_atual, bias_atual, ativacao="relu"):
    # cálculo da entrada para a função de ativação
    Saida_atual = np.dot(Pesos_atual, Ativado_anterior) + bias_atual

    # selecção da função de ativação
    if ativacao is "relu":
        func_ativacao = relu
    elif ativacao is "sigmoid":
        func_ativacao = sigmoid
    else:
        raise Exception('Ainda não implementamos essa funcao')

    # retorna a ativação calculada Ativado_atual e a matriz intermediária Saida
    return func_ativacao(Saida_atual), Saida_atual


# In[17]:


def propaga_total(X, valores_parametros, arquitetura):
    # memoria temporaria para a retropropagacao
    memoria = {}
    # O vetor X é a ativação para a camada 0 
    Ativado_atual = X

    # iterações para as camadas
    for indice, camada in enumerate(arquitetura):
        # a numeração das camadas começa de 1
        indice_camada = indice + 1
        # utiliza a ativação da iteração anterior
        Ativado_anterior = Ativado_atual

        # extrai a função de ativação para a camada atual
        func_ativacao_atual = camada["ativacao"]
        # extrai os pesos da camada atual
        Pesos_atual = valores_parametros["P" + str(indice_camada)]
        # extrai o bias para a camada atual
        b_atual = valores_parametros["b" + str(indice_camada)]
        # cálculo da ativação para a camada atual
        Ativado_atual, Saida_atual = propaga_uma_camada(Ativado_anterior, Pesos_atual, b_atual, func_ativacao_atual)

        # salca os valores calculados na memória
        memoria["A" + str(indice)] = Ativado_anterior
        memoria["Z" + str(indice_camada)] = Saida_atual

    # retorna o vetor predito e um dicionário contendo os valores intermediários
    return Ativado_atual, memoria


# ### Testando a rede

# In[18]:


# y_estimado[0,0]*ymax


# In[19]:


# y[0]*ymax


# ### Atualização dos pesos

# In[20]:


def atualiza(valores_parametros, gradidentes, arquitetura, taxa_aprendizagem):
    # iterações pelas camadas
    for indice_camada, camada in enumerate(arquitetura, 1):
        valores_parametros["P" + str(indice_camada)] -= taxa_aprendizagem * gradidentes["dP" + str(indice_camada)]
        valores_parametros["b" + str(indice_camada)] -= taxa_aprendizagem * gradidentes["db" + str(indice_camada)]

    return valores_parametros;


# ### Função de custo

# In[21]:


def valor_de_custo(Y_predito, Y):
    # numero_de_exemplos
    m = Y_predito.shape[1]

    custo = -1 / m * (np.dot(Y, np.log(Y_predito).T) + np.dot(1 - Y, np.log(1 - Y_predito).T))

    return np.squeeze(custo)


# ### Retropropagação

# In[22]:


def retropropagacao_total(Y_predito, Y, memoria, valores_parametros, arquitetura):
    gradientes = {}

    # numero de exemplos
    # m = Y.shape[1]
    # para garantir que os dois vetores tenham a mesma dimensão
    Y = Y.reshape(Y_predito.shape)

    # inicia o algoritmo de gradiente descendente
    dAtivado_anterior = - (np.divide(Y, Y_predito) - np.divide(1 - Y, 1 - Y_predito));

    for indice_camada_anterior, camada in reversed(list(enumerate(arquitetura))):
        indice_camada_atual = indice_camada_anterior + 1
        # Função de ativação para a camada atual

        funcao_ativao_atual = camada["ativacao"]

        dAtivado_atual = dAtivado_anterior

        Ativado_anterior = memoria["A" + str(indice_camada_anterior)]
        Saida_atual = memoria["Z" + str(indice_camada_atual)]

        Pesos_atual = valores_parametros["P" + str(indice_camada_atual)]
        bias_atual = valores_parametros["b" + str(indice_camada_atual)]

        dAtivado_anterior, dPesos_atual, db_atual = retropropagacao_uma_camada(
            dAtivado_atual,
            Pesos_atual,
            bias_atual,
            Saida_atual,
            Ativado_anterior,
            funcao_ativao_atual)

        gradientes["dP" + str(indice_camada_atual)] = dPesos_atual
        gradientes["db" + str(indice_camada_atual)] = db_atual

    return gradientes


# In[23]:


'''
def sigmoid_retro(dAtivado, Saida):
    sig = sigmoid(Saida)
    return dAtivado * sig * (1 - sig)

'''

sigmoid_retro = lambda dAtivado, Saida: dAtivado * sigmoid(Saida) * (1 - sigmoid(Saida))


def relu_retro(dAtivado, Saida):
    dSaida = np.array(dAtivado, copy=True)
    dSaida[Saida <= 0] = 0;
    return dSaida;


# In[24]:


def retropropagacao_uma_camada(dAtivado_atual, Pesos_atual, b_atual, Saida_atual, Ativado_anterior, ativacao="relu"):
    # número de exemplos
    m = Ativado_anterior.shape[1]

    # seleção função de ativação
    if ativacao is "relu":
        func_ativacao_retro = relu_retro
    elif ativacao is "sigmoid":
        func_ativacao_retro = sigmoid_retro
    else:
        raise Exception('Ainda não implementamos essa funcao')

    # derivada da função de ativação
    dSaida_atual = func_ativacao_retro(dAtivado_atual, Saida_atual)

    # derivada da matriz de Pesos
    dPesos_atual = np.dot(dSaida_atual, Ativado_anterior.T) / m
    # derivada do vetor b
    db_atual = np.sum(dSaida_atual, axis=1, keepdims=True) / m
    # derivada da matriz A_anterior
    dAtivado_anterior = np.dot(Pesos_atual.T, dSaida_atual)

    return dAtivado_anterior, dPesos_atual, db_atual


# ### Treinamento

# In[25]:


def treino(X, Y, X_teste, Y_teste, arquitetura, epocas, taxa_aprendizagem):
    # Inicia os parâmetros da rede neural
    valores_parametros = inicia_camadas(arquitetura, 2)
    # Listas que vão guardar o progresso da aprendizagem da rede
    historia_custo = []
    historia_custo_teste = []

    # Atualiza a cada época
    for i in range(epocas):
        # Propaga a rede - Foward propagation
        Y_predito, memoria = propaga_total(X, valores_parametros, arquitetura)

        Y_predito_teste, memoria2 = propaga_total(X_teste, valores_parametros,
                                                  arquitetura)

        # calcula as métricas e salva nas listas de história
        custo = valor_de_custo(Y_predito, Y)
        historia_custo.append(custo)
        custo_teste = valor_de_custo(Y_predito_teste, Y_teste)
        historia_custo_teste.append(custo_teste)

        # Retropropagação - Backpropagation
        gradientes = retropropagacao_total(Y_predito, Y, memoria,
                                           valores_parametros, arquitetura)
        # Atualiza os pesos
        valores_parametros = atualiza(valores_parametros, gradientes,
                                      arquitetura, taxa_aprendizagem)

        if (i % 50 == 0):
            print("Iteração: {:05} - custo: {:.5f} ".format(i, custo))

    return valores_parametros, historia_custo, historia_custo_teste


# In[26]:


from sklearn.model_selection import train_test_split

# In[27]:


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.43, random_state=42)


# In[28]:


def testar_arquiteturas(epocas=1000, taxa_aprendizagem=0.01):
    treinamentos = []

    for arquitetura in arquiteturas:
        # Treinamento
        valores_parametros, historia_custo, historia_custo_teste = treino(np.transpose(X_treino),
                                                                          np.transpose(
                                                                              y_treino.reshape((y_treino.shape[0], 1))),
                                                                          np.transpose(X_teste),
                                                                          np.transpose(
                                                                              y_teste.reshape((y_teste.shape[0], 1))),
                                                                          arquitetura,
                                                                          epocas,
                                                                          taxa_aprendizagem)

        treinamento = {}
        treinamento['parametros'] = valores_parametros
        treinamento['historia_custo'] = historia_custo
        treinamento['historia_custo_teste'] = historia_custo_teste

        treinamentos.append(treinamento)

    return treinamentos


# In[29]:


def plotar_funcao_custo(historia_custo, historia_custo_teste):
    plt.plot(historia_custo)
    plt.plot(historia_custo_teste, 'r')
    plt.legend(['Treinamento', 'Teste'])
    plt.ylabel('Custo')
    plt.xlabel('Épocas')
    plt.show()


treinamentos = testar_arquiteturas(epocas=20000, taxa_aprendizagem=0.05)

# In[30]:


plotar_funcao_custo(treinamentos[0]['historia_custo'], treinamentos[0]['historia_custo_teste'])

# In[31]:


plotar_funcao_custo(treinamentos[1]['historia_custo'], treinamentos[1]['historia_custo_teste'])

# ### Fazendo Previsões

# In[32]:


# Previsão arquitetura 1
Y_pred_1, _1 = propaga_total(np.transpose(X_teste), treinamentos[0]['parametros'], arquiteturas[0])
# Previsão arquitetura 2
Y_pred_2, _2 = propaga_total(np.transpose(X_teste), treinamentos[1]['parametros'], arquiteturas[1])


# In[34]:


def plotar_previsoes_by_temperatura(Y_predito, criterio_X):
    if criterio_X == 'temperatura':
        indice_X = 1
    if criterio_X == 'clima':
        indice_X = 0
        plt.rcParams.update({'font.size': 22})
        indice = [1, 2, 3]
        plt.xticks(indice, fontsize=14)

    plt.plot(np.transpose(X_teste)[indice_X], ymax * y_teste, '.')
    plt.plot(np.transpose(X_teste)[indice_X], ymax * Y_predito.reshape([-1, 1]), '.r')
    plt.legend(['Reais', 'Preditos'])
    plt.ylabel('bicicletas_alugadas')
    plt.xlabel(criterio_X)
    plt.show()


plotar_previsoes_by_temperatura(Y_pred_1, 'temperatura')

# In[35]:


plotar_previsoes_by_temperatura(Y_pred_1, 'clima')

# In[37]:


plotar_previsoes_by_temperatura(Y_pred_2, 'temperatura')

# In[38]:


plotar_previsoes_by_temperatura(Y_pred_2, 'clima')
