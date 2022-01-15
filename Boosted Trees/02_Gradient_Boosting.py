#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('mushrooms.csv')


# In[3]:


df.head()


# In[4]:


X = df.drop('class', axis=1)


# In[5]:


X = pd.get_dummies(X, drop_first=True)


# In[6]:


y = df['class']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[9]:


from sklearn.ensemble import GradientBoostingClassifier


# In[10]:


from sklearn.model_selection import GridSearchCV


# In[11]:


param_grid = {'n_estimators':[50,100], 'learning_rate': [0.1,0.05,0.2], 'max_depth':[3,4,5]}


# In[12]:


gb_model = GradientBoostingClassifier()


# In[14]:


grid = GridSearchCV(gb_model, param_grid)


# In[15]:


grid.fit(X_train,y_train)


# In[16]:


from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score


# In[17]:


pred = grid.predict(X_test)


# In[18]:


pred


# In[20]:


grid.best_params_


# In[21]:


print(classification_report(y_test, pred))


# In[23]:


feat_import = grid.best_estimator_.feature_importances_


# In[24]:


im_feat = pd.DataFrame(index=X.columns, data=feat_import, columns=['Importance'])


# In[29]:


im_feat = im_feat[im_feat['Importance']>0.0005].sort_values('Importance')


# In[30]:


im_feat


# In[35]:


plt.figure(figsize=(12,6))
sns.barplot(data=im_feat, x=im_feat.index, y='Importance')
plt.xticks(rotation=90);


# In[ ]:




