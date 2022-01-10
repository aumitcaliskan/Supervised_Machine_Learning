#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns


# In[2]:


df = pd.read_csv('penguins_size.csv')


# In[3]:


df.head()


# In[4]:


df = df.dropna()


# In[5]:


df.head()


# In[6]:


X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)


# In[7]:


y = df['species']


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


rfc = RandomForestClassifier(n_estimators=10, max_features='auto', random_state=101)


# In[12]:


rfc.fit(X_train, y_train)


# In[13]:


preds = rfc.predict(X_test)


# In[15]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


# In[16]:


plot_confusion_matrix(rfc, X_test, y_test)


# In[17]:


print(classification_report(y_test, preds))


# In[18]:


rfc.feature_importances_


# In[ ]:




