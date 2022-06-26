#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[5]:


kanser_data= load_breast_cancer()
X=kanser_data.data 
y=kanser_data.target


# In[6]:


X


# In[7]:


y


# In[8]:


X_train, X_test, y_train, y_test=train_test_split(X,y, 
                                                  test_size=0.2,
                                                  random_state=0)
classifier=RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train) #modeli eğitiyoruz


# In[9]:


predictions=classifier.predict(X_test) #y_pred


# In[10]:


y_test #gerçek


# In[11]:


predictions #tahmin


# In[12]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[13]:


#confusion matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confusion_matrix(y_test, predictions)


# In[14]:


import pandas as pd
pd.crosstab(y_test, predictions, rownames=['True'],
            colnames=['Predicted'], margins=True)


# In[15]:


import seaborn as sns
cnf_matrix=confusion_matrix(y_test, predictions)
p=sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[16]:


Accuracy_score=(TP+TN) / (TP+TN+FP+FN)
#(45+65)/45+65+2+2
#110 / 114
#0.9649.....
Accuracy_score


# In[19]:


from sklearn.metrics import recall_score
recall_score(y_test, predictions)


# In[17]:


TP / TP + FN
45 / 45 + 2 


# In[18]:


from sklearn.metrics import precision_score
precision_score(y_test, predictions)


# In[ ]:


TP / TP + FP


# In[17]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions)


# In[18]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predictions)


# In[21]:


import numpy as np
rmse=np.sqrt(mean_squared_error(y_test, predictions))
rmse


# In[26]:


#AUC - ROC Curve
from sklearn.metrics import roc_curve,auc
class_pr= classifier.predict_proba(X_test)
preds= class_pr[:,1]
fpr, tpr, threshold = roc_curve(y_test, predictions)
print('FPR',fpr)
print('TPR', tpr)
print('thresholds',threshold)

roc_auc=auc(fpr, tpr)
print(roc_auc)

plt.title('ROC') 
plt.plot(fpr, tpr, 'b', label='AUC %0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('FPR false positive rate')
plt.ylabel('TPR true positive rate')
plt.show()


# In[ ]:




