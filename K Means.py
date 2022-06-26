#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.datasets import load_iris 


# In[7]:


iris=load_iris()
x=pd.DataFrame(iris.data) 
x.columns=["sepal_length","sepal_width","petal_length","petal_width"]


# In[8]:


y= pd.DataFrame(iris.target)
y.columns=["target"]


# In[9]:


plt.figure(figsize = (14,7))
colormap = np.array(['red','blue','green'])
plt.subplot(1,2,1)
plt.scatter(x.sepal_length,x.sepal_width, c=colormap[y.target],
            s=40)
plt.title("sepal")
plt.subplot(1,2,2)
plt.scatter(x.petal_length,x.petal_width, c=colormap[y.target],
            s=40)
plt.title("petal")
plt.show()


# In[10]:


model=KMeans(n_clusters=3)
model.fit(x)


# In[5]:


plt.figure(figsize = (14,7))
colormap = np.array(['red','blue','green'])
plt.subplot(1,2,1)
plt.scatter(x.sepal_length,x.sepal_width, c=colormap[y.target],
            s=40)
plt.title("Gerçek Değerler")
plt.subplot(1,2,2)
plt.scatter(x.sepal_length,x.sepal_width, c=colormap[model.labels_],
            s=40)
plt.title("KMeans")
plt.show()


# In[29]:


print(model.labels_)
print(y.target)


# In[31]:


sm.accuracy_score(y.target,model.labels_)


# In[ ]:




