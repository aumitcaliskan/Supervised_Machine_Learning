#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('cement_slump.csv')


# In[3]:


df.head()


# In[4]:


plt.figure(figsize=(8,8), dpi=80)
sns.heatmap(df.corr(), annot=True)


# In[5]:


df.columns


# In[6]:


X = df.drop('Compressive Strength (28-day)(Mpa)', axis=1)


# In[7]:


y = df['Compressive Strength (28-day)(Mpa)']


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[12]:


scaled_X_train = scaler.fit_transform(X_train)


# In[13]:


scaled_X_test = scaler.transform(X_test)


# In[14]:


from sklearn.svm import SVR, LinearSVR


# In[15]:


base_model = SVR()


# In[16]:


base_model.fit(scaled_X_train, y_train)


# In[17]:


base_preds = base_model.predict(scaled_X_test)


# In[18]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[19]:


mean_absolute_error(y_test, base_preds)


# In[20]:


np.sqrt(mean_squared_error(y_test, base_preds))


# In[21]:


y_test.mean()


# In[22]:


param_grid = {'C':[0.001,0.01,0.1,0.5,1], 
              'kernel':['linear','rbf', 'poly'], 
              'gamma':['scale','auto'],
             'degree':[2,3,4],
             'epsilon':[0,0.01,0.1,0.5,1,2]}

# epsilon: error you are wiling to allow. 0 overfit


# In[23]:


from sklearn.model_selection import GridSearchCV


# In[24]:


svr = SVR()


# In[25]:


grid = GridSearchCV(svr, param_grid)


# In[26]:


grid.fit(scaled_X_train, y_train)


# In[27]:


grid.best_params_


# In[28]:


grid_preds = grid.predict(scaled_X_test)


# In[29]:


mean_absolute_error(y_test, grid_preds)


# In[30]:


np.sqrt(mean_squared_error(y_test,grid_preds))


# In[ ]:




