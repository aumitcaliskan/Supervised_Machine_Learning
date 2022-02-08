#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Scikit-learn is a library containig many ML algorithms. 
# "estimator API"
# test various approaches
# applying models and performance metrics


# In[2]:


# what is the relationship between each ad channel(TV, radio, newspaper) and sales?


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df =pd.read_csv('Advertising.csv') 


# In[5]:


df.head()


# In[6]:


fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))

axes[0].plot(df['TV'], df['sales'],'o')
axes[0].set_ylabel('sales')
axes[0].set_xlabel('TV spend')

axes[1].plot(df['radio'], df['sales'],'o')
axes[1].set_ylabel('sales')
axes[1].set_xlabel('radio')

axes[2].plot(df['newspaper'], df['sales'],'o')
axes[2].set_ylabel('sales')
axes[2].set_xlabel('newspaper')


# In[7]:


sns.pairplot(df,corner=True)


# In[8]:


X = df.drop('sales',axis=1)


# In[9]:


X


# In[10]:


y = df['sales']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


# help(train_test_split)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# test_size what percentage of data should go to test? generally people use 0.3
# random_state similar to random.seed. it is important to use for comparison between different algorithms. shuffle the data.


# In[14]:


len(df)


# In[15]:


X_train


# In[16]:


len(X_test)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


X_test.head()


# In[21]:


y_test.head()


# In[22]:


test_predictions = model.predict(X_test)


# In[23]:


test_predictions


# ## Performance Evaluation

# In[24]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[25]:


df['sales'].mean()


# In[26]:


sns.histplot(data=df, x='sales',bins=20)


# In[27]:


mean_absolute_error(y_test,test_predictions)

# df['sales'].mean() arasında yaklaşık yüzde 10 hata var


# In[28]:


mean_squared_error(y_test,test_predictions)

# I cannot compare this value with df['sales'].mean(). Because this is squared error. 


# In[29]:


np.sqrt(mean_squared_error(y_test,test_predictions))

# root mean squared error. Yaklaşık yüzde 10 hata var.


# ## Residual Plots

# In[30]:


test_residuals = y_test - test_predictions


# In[31]:


ax = sns.scatterplot(x=y_test, y=test_residuals)
ax.set_ylabel('residuals')
ax.set_xlabel('y_test')

plt.axhline(y=0, color='red', ls='--')


# In[32]:


# no clear line or curve. I should not use linear regression. looks random.Should look normal.


# In[33]:


sns.displot(test_residuals, bins=25, kde=True)


# In[34]:


# it doesn't look like a normal distribution.


# In[35]:


import scipy as sp


# In[36]:


fig, ax = plt.subplots(figsize=(6,8), dpi=100)

_ = sp.stats.probplot(test_residuals, plot=ax) 


# ## Model Deployment and Coefficient Interpretation

# In[37]:


final_model = LinearRegression()


# In[38]:


final_model.fit(X,y)

# fit to whole data set


# In[39]:


final_model.coef_


# In[43]:


X.head()

# 0.04576465 coef of TV, 0.18853002 coef of radio, -0.00103749 coef of newspaper

# coef of newspaper is almost zero. Model thinks that it is useless and even negative coef means it decreases the sales. 

# If you increase radio[0] 37.8 to 38.8 (1 unit) it increases sales 0.188 (coef) unit


# In[44]:


y_hat = final_model.predict(X)


# In[45]:


fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))

axes[0].plot(df['TV'], df['sales'],'o')
axes[0].plot(df['TV'], y_hat,'o', color='red')
axes[0].set_ylabel('sales')
axes[0].set_xlabel('TV spend')

axes[1].plot(df['radio'], df['sales'],'o')
axes[1].plot(df['radio'], y_hat,'o', color='red')
axes[1].set_ylabel('sales')
axes[1].set_xlabel('radio')

axes[2].plot(df['newspaper'], df['sales'],'o')
axes[2].plot(df['newspaper'], y_hat,'o', color='red')
axes[2].set_ylabel('sales')
axes[2].set_xlabel('newspaper')


# In[46]:


# comparing visually actual regression predictions versus true feature values


# In[47]:


# don't forget to normalize the feautures. They can have different units.


# ## Save and Load

# In[48]:


from joblib import dump,load


# In[51]:


dump(final_model, 'final_sales_model.joblib')

# to save the model to computer


# In[54]:


loaded_model = load('final_sales_model.joblib')

# to load the saved model


# In[55]:


loaded_model.coef_


# In[56]:


X.shape


# In[57]:


# example campaign should be two dimensional


# In[58]:


# 149 tv, 22 radio, 12 newspaper
# sales?

campaign = [[149,22,12]]


# In[60]:


loaded_model.predict(campaign)

# 13.89 unit sales


# In[ ]:




