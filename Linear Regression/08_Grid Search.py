#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Advertising.csv")


# In[3]:


df.head()


# In[4]:


X = df.drop('sales',axis=1)
y = df['sales']


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


from sklearn.linear_model import ElasticNet


# In[9]:


# help(ElasticNet)


# In[10]:


base_elastic_net_model = ElasticNet()


# In[22]:


param_grid = {'alpha': [0.1,1,5,10,50,100], 
              'l1_ratio':[.1,.5,.7,.95,.99,1]}

# can be changed to evaluate the model


# In[12]:


from sklearn.model_selection import GridSearchCV


# In[13]:


grid_model = GridSearchCV(estimator=base_elastic_net_model, 
                          param_grid=param_grid, 
                          scoring='neg_mean_squared_error', 
                          cv=5,verbose=2)


# In[14]:


grid_model.fit(X_train,y_train)


# In[15]:


grid_model.best_estimator_


# In[16]:


grid_model.best_params_


# In[18]:


pd.DataFrame(grid_model.cv_results_)


# In[19]:


y_pred = grid_model.predict(X_test)


# In[20]:


from sklearn.metrics import mean_squared_error


# In[21]:


mean_squared_error(y_test, y_pred)

