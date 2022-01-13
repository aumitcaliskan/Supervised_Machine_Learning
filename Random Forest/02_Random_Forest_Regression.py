#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('rock_density_xray.csv')


# In[4]:


df.head()


# In[5]:


df.columns = ['Signal', 'Density']


# In[7]:


sns.scatterplot(x='Signal', y='Density', data = df)


# In[15]:


X = df['Signal'].values.reshape(-1,1)
y = df['Density']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


lr_model = LinearRegression()


# In[20]:


lr_model.fit(X_train, y_train)


# In[25]:


lr_preds = lr_model.predict(X_test)


# In[26]:


lr_preds


# In[23]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[27]:


mean_absolute_error(y_test, lr_preds)


# In[28]:


np.sqrt(mean_squared_error(y_test, lr_preds))


# In[31]:


signal_range = np.arange(0,100)


# In[32]:


signal_preds = lr_model.predict(signal_range.reshape(-1,1))


# In[33]:


plt.figure(figsize=(12,8), dpi=200)
sns.scatterplot(x='Signal', y='Density', data = df)

plt.plot(signal_range, signal_preds)


# In[37]:


def run_model(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train,y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    
    signal_range = np.arange(0,100)
    signal_preds = model.predict(signal_range.reshape(-1,1))
    
    plt.figure(figsize=(12,8), dpi= 200)
    sns.scatterplot(x='Signal', y= 'Density',data =df, color='black')
    
    plt.plot(signal_range, signal_preds)


# In[38]:


model = LinearRegression()
run_model(model, X_train, y_train, X_test, y_test)


# In[39]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# In[50]:


pipe =make_pipeline(PolynomialFeatures(degree=6), LinearRegression())


# In[51]:


run_model(pipe, X_train, y_train, X_test, y_test)


# In[52]:


from sklearn.neighbors import KNeighborsRegressor


# In[56]:


k_values = [1,5,10,20,30]

for n in k_values:
    
    model = KNeighborsRegressor(n_neighbors=n)
    run_model (model, X_train, y_train, X_test, y_test)


# In[57]:


from sklearn.tree import DecisionTreeRegressor


# In[58]:


model =DecisionTreeRegressor()
run_model (model, X_train, y_train, X_test, y_test)


# In[59]:


from sklearn.svm import SVR


# In[60]:


from sklearn.model_selection import GridSearchCV


# In[61]:


svr =SVR()
param_grid = {'C' : [0.01, 0.1, 1,5,10,100,1000], 'gamma':['auto','scale']}

grid = GridSearchCV(svr, param_grid)


# In[62]:


run_model (grid, X_train, y_train, X_test, y_test)


# In[63]:


grid.best_params_


# In[64]:


from sklearn.ensemble import RandomForestRegressor


# In[65]:


rfr = RandomForestRegressor(n_estimators=10)


# In[66]:


run_model (rfr, X_train, y_train, X_test, y_test)


# In[67]:


from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor


# In[68]:


model = GradientBoostingRegressor()
run_model (model, X_train, y_train, X_test, y_test)


# In[69]:


model = AdaBoostRegressor()
run_model (model, X_train, y_train, X_test, y_test)


# In[ ]:




