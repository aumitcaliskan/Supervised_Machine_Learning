#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification


# In[25]:


X,y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=53)


# In[26]:


X.shape


# In[27]:


plt.scatter(X.T[0], X.T[1], c=y)


# In[28]:


model = LogisticRegression()


# In[29]:


model.fit(X,y)


# In[36]:


x11, x22 = np.meshgrid(
    np.linspace(X.T[0].min(), X.T[0].max(), 1000),
    np.linspace(X.T[1].min(), X.T[1].max(), 1000))


# In[37]:


x11.shape


# In[41]:


x11.ravel().shape


# In[42]:


plt.scatter(X.T[0], X.T[1], c=y, edgecolors='black')
plt.contour(x11, x22, model.predict(np.array([x11.ravel(), x22.ravel()]).T).reshape(x11.shape), alpha=0.1)


# ### Let'a use polynomial

# In[44]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate
from sklearn import preprocessing;


# In[45]:


model = LogisticRegression(solver= 'liblinear')


# In[46]:


poly = PolynomialFeatures(2)


# In[47]:


poly.fit(X)


# In[48]:


model.fit(poly.transform(X),y)


# In[51]:


for p in range(2,10):

    poly = PolynomialFeatures(p)
    poly.fit(X)
    score = cross_validate(model, preprocessing.scale(poly.transform(X)), y, cv=10)['test_score'].mean()
    ts = cross_validate(model, preprocessing.scale(poly.transform(X)), y, cv=10, return_train_score=True)['train_score'].mean()
    print(f"derecem: {p}, train score: {round(ts,3)}, test score: {round(score,3)}")


# In[52]:


model = LogisticRegression(solver= 'liblinear')


# In[53]:


poly = PolynomialFeatures(3)


# In[54]:


poly.fit(X)


# In[55]:


model.fit(poly.transform(X),y)


# In[56]:


plt.scatter(X.T[0], X.T[1], c=y, edgecolors='black')
plt.contourf(x11, x22, model.predict(poly.transform(np.array([x11.ravel(), x22.ravel()]).T)).reshape(x11.shape), alpha=0.3)

