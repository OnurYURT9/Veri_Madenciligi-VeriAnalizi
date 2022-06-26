#!/usr/bin/env python
# coding: utf-8

# In[1]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset _temizleme.xls',sheet_name='Training_Data',index_col=None)


# In[2]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset _temizleme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1,2,3,4])


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset _temizleme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1,2,3,4])


# In[5]:


trainDS.head()


# In[6]:


trainDS.tail(3)


# In[7]:


trainDS.shape


# In[8]:


trainDS["STR"].head()


# In[9]:


trainDS.dtypes


# In[10]:


trainDS["STR","STG"].head()


# In[11]:


eksik_veri= trainDS.isnull().sum() #girilmemiş veriler
print(eksik_veri)


# In[12]:


satir_olarak_ihmal = trainDS.dropna() #eksik olan satırları kaldırdık
satir_olarak_ihmal


# In[13]:


sutun_olarak_ihmal = trainDS.dropna(axis=1)
sutun_olarak_ihmal


# In[14]:


print(trainDS.shape)
print("satir",trainDS.shape[0])
print(sutun_olarak_ihmal.shape)
print(satir_olarak_ihmal.shape)


# In[15]:


#eksik verilere sabit değerler atamak için
sabit_deger=trainDS.fillna(0.0001)
sabit_deger.head()


# In[16]:


#son gözlemin aktarılması
son_gozlem=trainDS.fillna(method="ffill")
son_gozlem


# In[17]:


sonraki_gozlem=trainDS.fillna(method="bfill")
sonraki_gozlem


# In[18]:


from sklearn.preprocessing import Imputer


# In[ ]:




