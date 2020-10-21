
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando base de dados
base = pd.read_csv('census.csv')


# In[3]:


# Dividir entre previsores e classe
previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values


# In[4]:


# Transformando atributos nominais em atributos discretos
from sklearn.preprocessing import LabelEncoder


# In[6]:


labelencoder_previsores = LabelEncoder()


# In[7]:


previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])


# In[9]:


# Adicionando as variáveis dummies para as variáveis categóricas
from sklearn.preprocessing import OneHotEncoder


# In[10]:


onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()


# In[12]:


labelencoder_classe = LabelEncoder()


# In[13]:


classe = labelencoder_classe.fit_transform(classe)


# In[18]:


# Verificando números de linhas e colunas das colunas previsoras e da classe
print(previsores.shape)
print(classe.shape)


# In[20]:


# Escalonamento de vetores por padronização
from sklearn.preprocessing import StandardScaler


# In[21]:


scaler = StandardScaler()


# In[23]:


previsores = scaler.fit_transform(previsores)

