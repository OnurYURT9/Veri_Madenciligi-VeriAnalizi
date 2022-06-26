#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
data = pd.read_csv("a.txt",encoding='utf-8')


# In[6]:


data


# In[7]:


cumle= data.iloc[:,0]
cumle = [doc for doc in data.iloc[:,0]] #cümleleri grupladı
cumle


# In[8]:


#sinif= data.iloc[:,0]
sinif = [doc for doc in data.iloc[:,1]] #sonuc kısmını sınfladı
sinif


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


vektorzier = TfidfVectorizer(analyzer='word',lowercase=True)


# In[11]:


train = vektorzier.fit_transform(cumle)


# In[12]:


train.toarray()


# In[13]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# In[14]:


model = clf.fit(X=train.toarray(),y=sinif)


# In[15]:


test_verisi = vektorzier.transform(['bu kadar kötü ürün görmedim'])


# In[16]:


y_pred=model.predict(test_verisi.toarray())


# In[17]:


print(y_pred)


# In[20]:


test_verisi=vektorzier.transform(['iyi bir ürünmüş'])
print(test_verisi.toarray())


# In[ ]:




