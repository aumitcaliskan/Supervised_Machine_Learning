#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("Advertising.csv")


# In[3]:


df.head()


# In[21]:


X = df.drop('sales',axis=1)


# In[22]:


y = df['sales']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()


# In[27]:


scaler.fit(X_train)


# In[28]:


X_train = scaler.transform(X_train)


# In[29]:


X_test = scaler.transform(X_test)


# In[30]:


from sklearn.linear_model import Ridge


# In[31]:


model = Ridge(alpha=100)


# In[32]:


from sklearn.model_selection import cross_val_score


# In[33]:


scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)


# In[34]:


scores


# In[35]:


abs(scores.mean())


# In[36]:


model = Ridge(alpha=1)


# In[37]:


scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)


# In[38]:


abs(scores.mean())


# In[39]:


model.fit(X_train, y_train)


# In[40]:


y_final_test_pred = model.predict(X_test)


# In[41]:


from sklearn.metrics import mean_squared_error


# In[42]:


mean_squared_error(y_test, y_final_test_pred)

