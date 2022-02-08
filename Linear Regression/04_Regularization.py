#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('Advertising.csv')


# In[5]:


X = df.drop('sales',axis=1)


# In[7]:


y = df['sales']


# In[9]:


from sklearn.preprocessing import PolynomialFeatures


# In[10]:


polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)


# In[11]:


poly_features = polynomial_converter.fit_transform(X)


# In[12]:


X.shape


# In[13]:


poly_features.shape


# In[14]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.30, random_state=101)


# In[17]:


X_train.shape


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler = StandardScaler()


# In[20]:


scaler.fit(X_train)


# In[21]:


X_train = scaler.transform(X_train)


# In[22]:


X_test = scaler.transform(X_test)


# In[25]:


X_train[0]

# scaled down to pretty much same range


# In[26]:


poly_features[0]


# ## Ridge Regression

# In[27]:


from sklearn.linear_model import Ridge


# In[30]:


# help(Ridge)


# In[31]:


ridge_model = Ridge(alpha=10)


# In[32]:


ridge_model.fit(X_train,y_train)


# In[33]:


test_predictions = ridge_model.predict(X_test)


# In[34]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[35]:


MAE = mean_absolute_error(y_test, test_predictions)


# In[36]:


MAE


# In[37]:


RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))


# In[38]:


RMSE


# In[39]:


# we don't know alpha=10 is the best choice. Cross validation


# In[41]:


from sklearn.linear_model import RidgeCV

#Ridge Cross Validation


# In[42]:


ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0))

# cv=None to use the efficient Leave-One-Out cv. With big data set this can take time.


# In[43]:


ridge_cv_model.fit(X_train,y_train)


# In[45]:


ridge_cv_model.alpha_

#which alpha performs best


# In[46]:


from sklearn.metrics import SCORERS


# In[48]:


SCORERS.keys()

# higher is better


# In[49]:


ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')


# In[50]:


ridge_cv_model.fit(X_train,y_train)


# In[52]:


ridge_cv_model.alpha_


# In[53]:


test_predictions = ridge_cv_model.predict(X_test)


# In[54]:


MAE = mean_absolute_error(y_test,test_predictions)


# In[55]:


RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))


# In[56]:


MAE


# In[57]:


RMSE


# In[58]:


ridge_cv_model.coef_


# In[59]:


ridge_cv_model.best_score_


# ## Lasso Regression

# In[61]:


from sklearn.linear_model import LassoCV


# In[62]:


lasso_cv_model = LassoCV(eps=0.001, n_alphas=100, cv=5)


# In[64]:


lasso_cv_model.fit(X_train,y_train)

# Increase the max_iter to make alpha converge

# Increase eps


# In[86]:


lasso_cv_model = LassoCV(eps=0.001, n_alphas=100, cv=5, max_iter=1000000)

# Increase the max_iter to make alpha converge


# In[87]:


lasso_cv_model.fit(X_train,y_train)


# In[68]:


lasso_cv_model = LassoCV(eps=0.1, n_alphas=100, cv=5)

# Increase eps


# In[69]:


lasso_cv_model.fit(X_train,y_train)


# In[88]:


lasso_cv_model.alpha_


# In[71]:


test_predictions = lasso_cv_model.predict(X_test)


# In[72]:


MAE = mean_absolute_error(y_test,test_predictions)


# In[73]:


RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))


# In[74]:


MAE


# In[75]:


RMSE


# In[76]:


lasso_cv_model.coef_


# In[77]:


from sklearn.linear_model import ElasticNetCV


# In[78]:


elastic_model = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], eps=0.001, n_alphas=100, max_iter=1000000)

# Note that a good choice of list of values for l1_ratio is often to put more values close to 1 (i.e. Lasso) and 
# less close to 0 (i.e. Ridge), as in [.1, .5, .7, .9, .95, .99, 1].


# In[79]:


elastic_model.fit(X_train,y_train)


# In[82]:


elastic_model.l1_ratio_

# decided lasso


# In[83]:


elastic_model.alpha_


# In[89]:


lasso_cv_model.alpha_


# In[90]:


test_predictions = elastic_model.predict(X_test)


# In[92]:


MAE = mean_absolute_error(y_test,test_predictions)


# In[93]:


MAE

