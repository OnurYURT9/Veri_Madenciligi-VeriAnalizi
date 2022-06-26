#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


iris=load_iris()


# In[3]:


iris.data


# In[4]:


iris.target


# In[14]:


#import matplotlib.pyplot as plt
#import numpy as np

plt.figure(figsize=(2,1))
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target, 
            s=iris.data[:,2]*20)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()


# In[16]:


colors=["r","g","b"]
plt.figure(figsize=(10,6))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

for i,name in enumerate(iris.target_names):
    plt.scatter(iris.data[iris.target==i,0], 
                iris.data[iris.target==i,1], 
                iris.data[iris.target==i,2], 
                c=colors[i])

plt.show()


# In[ ]:




