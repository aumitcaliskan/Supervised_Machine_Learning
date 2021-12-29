#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df['species'].value_counts()


# In[8]:


sns.countplot(x='species', data=df)

# to check the data balanced or imbalanced


# In[10]:


sns.scatterplot(x='petal_length', y='petal_width', data=df, hue='species')


# In[11]:


sns.pairplot(df,hue='species')


# In[15]:


sns.heatmap(df.corr(),annot=True)


# In[16]:


X = df.drop('species',axis=1)


# In[19]:


y = df['species']

# don't need to code species as integers. sklearn doesn't have problem passing classes as string.


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


from sklearn.preprocessing import MinMaxScaler


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[23]:


scaler = MinMaxScaler()


# In[24]:


scaled_X_train = scaler.fit_transform(X_train)


# In[25]:


scaled_X_test = scaler.transform(X_test)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


from sklearn.model_selection import GridSearchCV


# penalty = l1,l2 or elasticnet. If we use elasticnet we have to use l1_ratio= 0 < float < 1
# 
# multi_class (default=auto)
# 
#     * ovr : one vs all classification
#         
#     * multinomial 
#  
#     ‘multinomial’ is unavailable when solver=’liblinear’. 
#     ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
#     
# solver
# 
#     * For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
# 
#     * For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
# 
#     * ‘liblinear’ is limited to one-versus-rest schemes.
#     
# Warning The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:
# 
#     * ‘newton-cg’ - [‘l2’, ‘none’]
# 
#     * ‘lbfgs’ - [‘l2’, ‘none’]
# 
#     * ‘liblinear’ - [‘l1’, ‘l2’]
# 
#     * ‘sag’ - [‘l2’, ‘none’]
# 
#     * ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, ‘none’]
#     
#   C = logarithmically 

# In[28]:


log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000 )


# In[29]:


penalty = ['l1','l2','elasticnet']
l1_ratio = np.linspace(0,1,20)
C = np.logspace(0,10,20)

param_grid = {'penalty':penalty, 'l1_ratio':l1_ratio,'C':C}


# In[30]:


grid_model = GridSearchCV(log_model,param_grid=param_grid)


# In[31]:


grid_model.fit(scaled_X_train, y_train)


# In[32]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# In[33]:


grid_model.best_params_ 


# In[34]:


y_pred = grid_model.predict(scaled_X_test)


# In[35]:


y_pred


# In[37]:


accuracy_score(y_test, y_pred)


# In[38]:


confusion_matrix(y_test,y_pred)


# In[39]:


plot_confusion_matrix(grid_model, scaled_X_test, y_test)


# In[40]:


print(classification_report(y_test, y_pred))


# In[41]:


from sklearn.metrics import plot_roc_curve


# In[44]:


plot_roc_curve(grid_model,scaled_X_test,y_test)

# we can't use plot_roc_curve for multiclass data


# In[45]:


# from sklearn documentation we need to check "Plot ROC curves for the multiclass problem"


# In[46]:


from sklearn.metrics import roc_curve, auc


# In[47]:


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


# In[48]:


plot_multiclass_roc(grid_model,scaled_X_test,y_test,n_classes=3)


# In[ ]:




