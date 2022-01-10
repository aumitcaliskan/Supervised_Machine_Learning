#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('penguins_size.csv')


# In[3]:


df.head()


# In[4]:


df['species'].unique()


# In[5]:


df['island'].unique()


# In[6]:


df['sex'].unique()


# In[7]:


df[df['sex']== '.']


# In[8]:


df[df['species'] == 'Gentoo'].groupby('sex').describe().T


# In[9]:


df.at[336,'sex'] = 'FEMALE'


# In[10]:


df.loc[336]


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df = df.dropna()


# In[14]:


df.info()


# In[15]:


sns.pairplot(df,hue='species')


# In[16]:


sns.catplot(x='species', y='culmen_length_mm', data=df, kind='box', col='sex')


# In[17]:


X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)


# In[18]:


y = df['species']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


model = DecisionTreeClassifier()


# In[23]:


model.fit(X_train, y_train)


# In[24]:


base_preds = model.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report,plot_confusion_matrix


# In[26]:


print(classification_report(y_test, base_preds))


# In[27]:


plot_confusion_matrix(model,X_test, y_test)


# In[28]:


model.feature_importances_


# In[29]:


X.columns


# In[30]:


pd.DataFrame(index=X.columns, data=model.feature_importances_, columns= ['Feature Importance']).sort_values('Feature Importance')


# In[31]:


from sklearn.tree import plot_tree


# In[32]:


plt.figure(figsize=(12,8),dpi=200)
plot_tree(model,feature_names=X.columns, filled=True);


# In[33]:


len(X_train)


# In[34]:


def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test, model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=200)
    plot_tree(model,feature_names=X.columns, filled=True);


# In[35]:


report_model(model)


# ### Hyperparameters

# In[36]:


pruned_tree = DecisionTreeClassifier(max_depth=3)


# In[37]:


pruned_tree.fit (X_train,y_train)


# In[38]:


report_model(pruned_tree)


# In[39]:


max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)


# In[40]:


max_leaf_tree.fit(X_train, y_train)


# In[41]:


report_model(max_leaf_tree)


# In[42]:


entropy_tree = DecisionTreeClassifier(criterion='entropy')


# In[43]:


entropy_tree.fit(X_train, y_train)


# In[44]:


report_model(entropy_tree)


# In[45]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


DecisionTreeClassifier()


# In[68]:


parameters = {"criterion":['gini','entropy'],
              "max_depth": range(1,20,2),
              "min_samples_leaf":range(1,20,2)
             }
grid_forest = GridSearchCV(estimator=model,
                      param_grid= parameters,
                      cv=3,return_train_score=True,
                      n_jobs= -1
                        )


# In[69]:


grid_forest.fit(X_train, y_train)


# In[73]:


df = pd.DataFrame(grid_forest.cv_results_)


# In[81]:


plt.figure(figsize=(12,8))
sns.heatmap(data=df[df['param_criterion'] == 'gini'].pivot_table(index='param_max_depth',columns= 'param_min_samples_leaf', values='mean_test_score'))


# In[77]:


df[df['param_criterion'] == 'gini'].pivot_table(index='param_max_depth',columns= 'param_min_samples_leaf', values='mean_test_score')


# In[71]:


grid_forest.best_params_


# In[65]:


grid_forest.best_score_


# In[82]:


parameters = {"max_depth": range(1,31)}

grid = GridSearchCV(estimator=model,
                      param_grid= parameters,
                      cv=3,return_train_score=True,
                      n_jobs= -1
                        )


# In[83]:


grid.fit(X_train, y_train)


# In[84]:


grid.best_params_


# In[85]:


df = pd.DataFrame(grid.cv_results_)


# In[88]:


df.set_index('param_max_depth')[['mean_train_score', 'mean_test_score']].plot()


# In[ ]:




