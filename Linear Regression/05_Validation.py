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


# ## Train | Test Split Procedure 
# 
# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Test for both X and y
# 2. Fit/Train Scaler on Training X Data
# 3. Scale X Test Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6

# In[4]:


X = df.drop('sales', axis=1)


# In[5]:


y = df['sales']


# ### Split Data

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Scale X Test Data

# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


scaler = StandardScaler()


# In[10]:


scaler.fit(X_train)


# In[11]:


X_train = scaler.transform(X_train)


# In[12]:


X_test = scaler.transform(X_test)


# ### Create Model

# In[13]:


from sklearn.linear_model import Ridge


# In[14]:


model = Ridge(alpha = 100)


# In[15]:


model.fit(X_train,y_train)


# In[16]:


y_pred = model.predict(X_test)


# ### Evaluate Model

# In[17]:


from sklearn.metrics import mean_squared_error


# In[18]:


mean_squared_error(y_test, y_pred)


# In[19]:


# if there is a better value for alpha?


# ### Adjust Parameters

# In[20]:


model_one = Ridge(alpha=1)


# In[21]:


model_one.fit(X_train,y_train)


# In[22]:


y_pred_two = model_one.predict(X_test)


# In[23]:


mean_squared_error(y_test,y_pred_two)


# In[24]:


# this alpha is better


# In[25]:


for i in range(1,11):
    model = Ridge(alpha=i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    print (mse)


# ## Train | Validation | Test Split Procedure 
# 
# This is often also called a "hold-out" set, since you should not adjust parameters based on the final test set, but instead use it *only* for reporting final expected performance.
# 
# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Validation/Test for both X and y
# 2. Fit/Train Scaler on Training X Data
# 3. Scale X Eval Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Evaluation Data (by creating predictions and comparing to Y_eval)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6
# 8. Get final metrics on Test set (not allowed to go back and adjust after this!)

# * Andrew NG göre genel olarak datayı(100) 80 train, 20 test olarak ayırmak lazım. 
# * Daha sonra train kısmını da aynı şekilde 60 train, 20 validation olarak ayırmak lazım.
# * Bu şekilde öncelikle train validation kısmında modeli geliştirip daha sonra teste sokmak daha sağlıklı
# * Son teste soktuktan sonra aldığımız hata son hata.

# In[26]:


X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)

#  X_train= 0.7 X_other(test)= 0.3 olarak ayrıldı


# In[27]:


X_eval, X_test, y_eval, y_test = train_test_split(X_other,y_other, test_size= 0.5, random_state=101)

# X_other(test) tekrar ayırdık. X_train = 0.7, X_eval = 0.15, X_test = 0.15


# In[28]:


scaler = StandardScaler()


# In[29]:


scaler.fit(X_train)


# In[30]:


X_train = scaler.transform(X_train)


# In[31]:


X_test = scaler.transform(X_test)


# In[32]:


X_eval = scaler.transform(X_eval)


# ### Create Model

# In[33]:


model_two = Ridge(alpha=100)


# In[34]:


model_two.fit(X_train,y_train)


# In[35]:


y_eval_pred = model_two.predict(X_eval)


# In[36]:


mean_squared_error(y_eval,y_eval_pred)


# ### Adjust Parameters and Re-evaluate

# In[39]:


model_three = Ridge(alpha=1)


# In[40]:


model_three.fit(X_train, y_train)


# In[41]:


new_y_eval_pred = model_three.predict(X_eval)


# In[44]:


mean_squared_error(y_eval,new_y_eval_pred)


# ### Final Evaluation (Can no longer edit parameters after this!)

# In[47]:


y_final_test_pred = model_three.predict(X_test)


# In[48]:


mean_squared_error(y_test, y_final_test_pred)

