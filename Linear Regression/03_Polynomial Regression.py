#!/usr/bin/env python
# coding: utf-8

# ## Training and Evaluation

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


X = df.drop('sales', axis=1)


# In[5]:


y = df['sales']


# In[6]:


from sklearn.preprocessing import PolynomialFeatures


# In[7]:


polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)


# In[8]:


polynomial_converter.fit(X)


# In[9]:


poly_features = polynomial_converter.transform(X)


# In[10]:


polynomial_converter.transform(X).shape


# In[11]:


X.shape


# In[12]:


X.iloc[0]


# In[13]:


poly_features[0]

# first 3 are the original. Three of them are interaction terms. 230.1*37.8 = 8697. Three of them are squarred terms. 


# In[14]:


polynomial_converter.fit_transform(X)

# fit and transform steps are together


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model.fit(X_train, y_train)


# In[20]:


test_predictions = model.predict(X_test)


# In[21]:


model.coef_


# In[22]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[23]:


MAE = mean_absolute_error(y_test,test_predictions)


# In[24]:


MSE = mean_squared_error(y_test,test_predictions)


# In[25]:


RMSE = np.sqrt(MSE)


# In[26]:


MAE


# In[27]:


RMSE


# In[28]:


# polynomial is better than simple inear.


# In[29]:


model.coef_


# In[30]:


poly_features[0]


# In[31]:


X.iloc[0]


# ## Choosing Degree

# In[32]:


# create the different order poly
# split poly feat train/test
# fit on train
# store/save the rmse for BOTH the train AND test
# PLOT teh results (error vs poly order)


# In[33]:


train_rmse_errors = []
test_rmse_errors = []

for d in range(1,10):
    
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly_converter.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))
    
    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)
    


# In[34]:


train_rmse_errors


# In[35]:


test_rmse_errors


# In[36]:


plt.plot(range(1,6),train_rmse_errors[:5],label='Train RMSE')
plt.plot(range(1,6),test_rmse_errors[:5],label='Test RMSE')

plt.xlabel('Degree of Poly')
plt.ylabel('RMSE')
plt.legend()


# In[37]:


plt.plot(range(1,10),train_rmse_errors,label='Train RMSE')
plt.plot(range(1,10),test_rmse_errors,label='Test RMSE')

plt.xlabel('Degree of Poly')
plt.ylabel('RMSE')
plt.legend()


# In[38]:


# we should use 2nd or 3rd degree of poly. 4th is riskly for complexity  


# ## Model Deployment

# In[39]:


final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)


# In[40]:


final_model = LinearRegression()


# In[42]:


full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X,y)


# In[43]:


from joblib import dump,load


# In[44]:


dump(final_model,"final_poly.joblib")


# In[45]:


dump(final_poly_converter,"final_converter.joblib")


# In[46]:


loaded_converter = load('final_converter.joblib')


# In[47]:


loaded_model = load('final_poly.joblib')


# In[48]:


campaign = [[149,22,12]]


# In[52]:


transformed_data = loaded_converter.fit_transform(campaign)


# In[53]:


loaded_model.predict(transformed_data)

