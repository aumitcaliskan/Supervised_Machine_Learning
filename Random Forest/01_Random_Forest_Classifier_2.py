#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('data_banknote_authentication.csv')


# In[3]:


df.head()


# In[4]:


sns.pairplot(df, hue='Class')


# In[5]:


X = df.drop('Class' , axis=1)


# In[6]:


y = df['Class']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[9]:


from sklearn.model_selection import GridSearchCV


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[12]:


n_estimators = [64,100,128,200]
max_features = [2,3,4]
bootstrap = [True, False]
oob_score = [True, False]


# In[13]:


param_grid = {'n_estimators':n_estimators, 'max_features':max_features, 'bootstrap':bootstrap, 'oob_score':oob_score}


# In[14]:


rfc = RandomForestClassifier()


# In[15]:


grid = GridSearchCV(rfc, param_grid)


# In[16]:


grid.fit(X_train, y_train)


# In[17]:


grid.best_params_


# In[18]:


rfc = RandomForestClassifier(max_features=2, n_estimators=64, oob_score=True)


# In[19]:


rfc.fit(X_train, y_train)


# In[20]:


rfc.oob_score_


# In[23]:


predictions = rfc.predict(X_test)


# In[27]:


from sklearn.metrics import plot_confusion_matrix,classification_report, accuracy_score


# In[25]:


print(classification_report(y_test, predictions))


# In[26]:


plot_confusion_matrix(rfc, X_test, y_test)


# In[28]:


errors = []
misclassifications = []

for n in range(1,200):
    
    rfc = RandomForestClassifier(n_estimators=n, max_features=2)
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    err = 1 - accuracy_score(y_test, preds)
    n_missed = np.sum(preds != y_test)
    
    errors.append(err)
    misclassifications.append(n_missed)


# In[29]:


plt.plot(range(1,200), errors)


# In[30]:


plt.plot(range(1,200), misclassifications)


# In[ ]:




