#!/usr/bin/env python
# coding: utf-8

# ## Data
# 
# An experiment was conducted on 5000 participants to study the effects of age and physical health on hearing loss, specifically the ability to hear high pitched tones. This data displays the result of the study in which participants were evaluated and scored for physical ability and then had to take an audio test (pass/no pass) which evaluated their ability to hear high frequencies. The age of the user was also noted. Is it possible to build a model that would predict someone's likelihood to hear the high frequency sound based solely on their features (age and physical score)?
# 
# * Features
# 
#     * age - Age of participant in years
#     * physical_score - Score achieved during physical exam
# 
# * Label/Target
# 
#     * test_result - 0 if no pass, 1 if test passed

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("hearing_test.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df['test_result'].value_counts()


# In[6]:


sns.countplot(data=df, x=df['test_result'])


# In[7]:


plt.figure(dpi=100)
sns.boxplot(x='test_result', y='age', data=df)


# In[8]:


plt.figure(dpi=100)
sns.boxplot(x='test_result', y='physical_score', data=df)


# In[9]:


plt.figure(dpi=100)
sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result', alpha= 0.5)


# In[10]:


sns.pairplot(df,hue='test_result')


# In[11]:


sns.heatmap(df.corr(),annot=True)


# In[12]:


sns.scatterplot(x='physical_score', y='test_result', data=df)

# boxplot is better


# ### 3D Scatterplot

# In[13]:


from mpl_toolkits.mplot3d import Axes3D


# In[14]:


fig = plt.figure(dpi=150)
ax = fig.add_subplot(projection='3d')
ax.scatter(df['age'], df['physical_score'], df['test_result'], c=df['test_result'] )


# In[15]:


df.head()


# In[16]:


X = df.drop('test_result', axis=1)


# In[17]:


y = df['test_result']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[21]:


scaler = StandardScaler()


# In[22]:


scaled_X_train = scaler.fit_transform(X_train)


# In[57]:


scaled_X_test = scaler.transform(X_test)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


log_model = LogisticRegression()


# In[26]:


log_model.fit(scaled_X_train,y_train)


# In[33]:


log_model.coef_

# physical score is much stronger predicter than age


# In[32]:


df.head()


# In[58]:


y_pred = log_model.predict_proba(scaled_X_test)


# In[59]:


y_pred


# In[60]:


y_pred = log_model.predict(scaled_X_test)


# In[61]:


y_pred


# ## Accuracy Score

# **Accuracy :** How often is the model correct?
#     
#     Acc = TP+TN / Total
#     
# **Accuracy paradox :** Imbalanced classses will always result in a distorted accuracy reflecting better performance than what is truly warranted
# 
#     * Medical conditions
#     * Fraud is not common
#     
# This means we shouldn't solely rely on accuracy as a metric!

# ## Recall Score

# **Recall :** When it actually is a positive case, how often is it correct?
#     
#     Recall = TP / Total Actual Positives
#     
# A recall of 0 alerts you the model isn't catching cases. Model is overfitted!

# ## Precision Score

# **Precision :** When prediction is positive, how often is it correct?
#     
#     Precision = TP / Total Predicted Positives

# Recall and Precision can help illuminate our performance specifically in regards to the relevant or positive case

# ## F1 Score (F Score)

# Harmonic mean of precision and recall. Since precision and recall are related to each other through TP
# 
#     F = 2*precision*recall / (precision+recall)
#     
# Harmonic mean allows the entire harmonic mean to go to zero if either precision or recall ends up being zero.

# ## ROC Curves

# There is a trade-off between True Positives and False Positives
# 
# In certain situations, we gladly accept more false positives to reduce false negatives. Ex: a dangerous virus test
#     
#     * By changing the cut-off limit we can change True vs False Positives
#     * Perfect model would have a zero FPR (False Positive Rate).
#     * AUC (Area Under the Curve): Allows us to compare different models. 1 is perfect
#     
# We can also create precision vs recall curves

# In[34]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[38]:


accuracy_score(y_test,y_pred)


# In[39]:


confusion_matrix(y_test,y_pred)


# In[40]:


from sklearn.metrics import plot_confusion_matrix


# In[62]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[46]:


classification_report(y_test,y_pred)


# In[47]:


print(classification_report(y_test,y_pred))


# In[50]:


# if precision and recall scores for classes(0,1) are close accuracy, you don't have an issue about imbalanced data.


# In[51]:


from sklearn.metrics import precision_score,recall_score


# In[52]:


precision_score(y_test,y_pred)


# In[53]:


recall_score(y_test,y_pred)


# In[55]:


from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve


# In[67]:


fig,ax = plt.subplots(figsize=(8,6))
plot_roc_curve(log_model,scaled_X_test, y_test, ax=ax)


# In[68]:


plot_precision_recall_curve(log_model,scaled_X_test,y_test)


# In[70]:


log_model.predict_proba(scaled_X_test)[0]


# In[71]:


y_test[0]


# In[ ]:




