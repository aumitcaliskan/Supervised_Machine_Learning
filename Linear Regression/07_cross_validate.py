#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("Advertising.csv")


# In[3]:


df.head()


# In[4]:


X = df.drop('sales',axis=1)


# In[5]:


y = df['sales']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(X_train)


# In[13]:


X_train = scaler.transform(X_train)


# In[14]:


X_test = scaler.transform(X_test)


# In[15]:


from sklearn.model_selection import cross_validate


# In[17]:


from sklearn.linear_model import Ridge


# In[18]:


model = Ridge(alpha=100)


# In[19]:


scores = cross_validate(model, X_train, y_train, scoring= ['neg_mean_squared_error', 'neg_mean_absolute_error'], cv=10)


# In[20]:


scores = pd.DataFrame(scores)


# In[21]:


scores


# In[22]:


scores.mean()


# In[23]:


model = Ridge(alpha=1)


# In[24]:


scores = cross_validate(model, X_train, y_train, scoring= ['neg_mean_squared_error', 'neg_mean_absolute_error'], cv=10)


# In[25]:


scores = pd.DataFrame(scores)


# In[26]:


scores


# In[27]:


scores.mean()


# In[28]:


model.fit(X_train, y_train)


# In[29]:


y_final_pred = model.predict(X_test)


# In[30]:


from sklearn.metrics import mean_squared_error   


# In[31]:


mean_squared_error(y_test, y_final_pred)

