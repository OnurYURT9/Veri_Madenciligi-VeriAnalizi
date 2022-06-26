#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


trainDS = pd.read_excel('Data_User_Modeling_Dataset_azaltma.xls',sheet_name='Training_Data',index_col=None,usecols=[0,1,2,3,4])
print(trainDS)


# In[5]:


trainDS.head()


# In[6]:


korelasyon_matris = trainDS.corr()
korelasyon_matris


# In[7]:


korelasyon_matris = trainDS.corr()
korelasyon_matris


# In[8]:


import seaborn as sns
sns.heatmap(korelasyon_matris,cmap="Reds")


# In[9]:


print(trainDS.columns[1])
gereksiz_veri=trainDS.drop(trainDS.columns[1],axis=1)
gereksiz_veri.head()


# In[17]:


varyans=trainDS.var()
print(varyans)


# In[10]:


varyans_filtresi= 0.0001
for i in range(0,len(varyans)):
    if(varyans[i]>=varyans_filtresi):
        nitelikler.append(trainDS.columns[i])
print(nitelikler)


# In[1]:


from sklearn.feature_selection import Variance


# In[ ]:




