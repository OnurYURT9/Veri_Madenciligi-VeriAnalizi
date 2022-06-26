#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[8]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset _birlestirme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1,2,3,4])
print(trainDS)
trainDS2 = pd.read_excel('Data_User_Modeling_Dataset _birlestirme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1])
print(trainDS2)


# In[7]:


print(trainDS.head())
print(trainDS2.head())


# In[7]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset_birlestirme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1,2,3,4])
print(trainDS)
trainDS2 = pd.read_excel('Data_User_Modeling_Dataset_birlestirme.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1])
print(trainDS2)


# In[8]:


print(trainDS.head())
print(trainDS2.head())


# In[1]:


birlestir1=pd.concat([trainDS,trainDS2])
print(birlestir1)


# In[11]:


birlestir2=pd.merge(trainDS,trainDS2,on="ID",how="left")
print(birlestir2)


# In[1]:


testDS = pd.read_excel('Data_User_Modeling_Dataset_birlestirme.xls',
                      sheet_name='Test_Data',
                      index_col=None, )


# In[ ]:




