#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score,accuracy_score, classification_report


# In[2]:


X, y = make_classification(n_samples =  1000, n_features=4, n_redundant=0, n_classes=2, random_state=53)


# In[3]:


X.shape


# In[4]:


y.shape


# In[5]:


model = LogisticRegression()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


X_train.shape


# In[8]:


X_test.shape


# In[9]:


model.fit(X_train,y_train)


# In[10]:


y_pred = model.predict(X_test)


# In[11]:


confusion_matrix(y_test, y_pred)


# In[12]:


recall_score(y_test,y_pred)


# In[13]:


precision_score(y_test,y_pred)


# In[14]:


accuracy_score(y_test,y_pred)


# In[15]:


f1_score(y_test,y_pred)


# In[16]:


print(classification_report(y_test,y_pred))

