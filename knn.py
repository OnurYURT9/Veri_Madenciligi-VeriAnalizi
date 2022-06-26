#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[38]:


weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast',
       'Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
     'Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes',
     'No','Yes','Yes','Yes','Yes','Yes','No']


# In[39]:


_label=preprocessing.LabelEncoder() # sayısal değerlere dönüştürülür alfabetik sıraya göre
weather_encoded=_label.fit_transform(weather)
print(weather_encoded)


# In[40]:


_label=preprocessing.LabelEncoder()
temp_encoded=_label.fit_transform(temp)
print(temp_encoded)


# In[41]:


features=list(zip(weather_encoded,temp_encoded)) 
features


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


play_encoded = _label.fit_transform(play)
print(play_encoded)


# In[44]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(features,play_encoded)


# In[45]:


predicted=model.predict([[0,2]]) #sınıf belirleme yes no mu?
print(predicted)


# In[ ]:





# In[ ]:





# In[80]:


from sklearn import datasets
wine=datasets.load_wine()


# In[81]:


print(wine.feature_names)
print(wine.target_names)


# In[82]:


print(wine.data[0:5])


# In[83]:


print(wine.target)


# In[84]:


print(wine.data.shape)
print(wine.target.shape)


# In[85]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=25,metric="manhattan") #model oluşturma
knn.fit(X_train,y_train) #modeli eğittimiz kısım
y_pred = knn.predict(X_test) #tahmin yapma
print(y_pred)
print(y_test)

from sklearn import metrics
print("Accuracy= ",metrics.accuracy_score(y_test,y_pred))


# In[ ]:




