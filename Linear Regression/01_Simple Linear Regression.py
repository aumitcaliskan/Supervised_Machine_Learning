#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Advertising.csv")


# In[4]:


df.head()


# In[8]:


df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']

# feature engineering. because total_spend didn't exist in the original data.


# In[9]:


df.head()


# In[10]:


sns.scatterplot(data=df, x='total_spend', y= 'sales')


# In[21]:


sns.regplot(data=df, x='total_spend', y= 'sales', line_kws = {'color': 'red'})


# In[22]:


X = df['total_spend']
y = df['sales']


# In[24]:


# y = mx + b
# y = B1x + B0
# help(np.polyfit)


# In[25]:


np.polyfit(X,y,deg=1)


# In[26]:


# B1 = 0.04 , B0 = 4.24


# In[27]:


potential_spend = np.linspace(0,500,100)


# In[28]:


predicted_sales = 0.04868788*potential_spend + 4.24302822


# In[31]:


sns.scatterplot(x='total_spend', y='sales', data=df)
plt.plot(potential_spend, predicted_sales, color='red')


# In[34]:


spend = 200

predicted_sales = 0.04868788*spend + 4.24302822


# In[33]:


predicted_sales


# In[36]:


np.polyfit(X,y,3)

# y = B3x**3 + B2x**2 + B1x + B0 higher order


# In[37]:


pot_spend = np.linspace(0,500,100)


# In[38]:


pred_sales = 3.07615033e-07*pot_spend**3 + -1.89392449e-04*pot_spend**2 + 8.20886302e-02*pot_spend + 2.70495053e+00


# In[45]:


sns.scatterplot(x='total_spend',y='sales',data=df)
plt.plot(pot_spend, pred_sales, color='red')
