#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  //
import seaborn as sns


# In[3]:


diabet=pd.read_csv('diabetes.csv')


# In[4]:


print(diabet.columns)


# In[7]:


print(diabet.shape) 
print(diabet.groupby('Outcome').size())


# In[8]:


diabet.hist(figsize=(9,9))


# In[9]:


diabet.groupby('Outcome').hist(figsize=(9,9))


# In[10]:


diabet_mod= diabet[(diabet.BloodPressure!=0) &
                  (diabet.BMI!=0) &
                  (diabet.Glucose!=0)]


# In[11]:


diabet_mod.shape


# In[1]:


feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 
               'SkinThickness', 'Insulin','BMI', 
               'DiabetesPedigreeFunction', 'Age']


# In[15]:


X=diabet_mod[feature_names]
y=diabet_mod.Outcome


# In[16]:


X


# In[17]:


y


# In[19]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# In[20]:


model=[]
model.append('DT', DecisionTreeClassifier())
model.append('GNB', GaussianNB())


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   stratify= diabet_mod.Outcome,
                                                   random_state=0)


# In[24]:


names=[]
scores=[]

for name, _model in model:   #modeldeki her isim için bunu tekrar yap
    _model.fit(X_train, y_train)   #x_train ve y_train i birbiri ile bağla modele uyarla
    y_pred = _model.predict(X_test)  #x_test e göre y tahmini yapar
    scores.append(accuracy_score(y_test, y_pred)) #tahmin ve test arasındaki gerçeklik
    names.append(name)

print(names)
print(scores)


# In[25]:


k_fold = StratifiedKFold(n_splits=5,random_state=10)
names=[]
scores=[]
for name, _model in model:
    score= cross_val_score(_model, X, y, cv=k_fold,  #bütün testlerin accuracy ortalaması
                           scoring='accuracy').mean()
    names.append(name)
    scores.append(score)

print(names)
print(scores)


# In[ ]:




