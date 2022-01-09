#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('mouse_viral_study.csv')


# In[3]:


df.head()


# In[4]:


df['Virus Present'].value_counts()


# In[5]:


sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present',data=df)

# hyperplane(2d line)

x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b

plt.plot(x,y, 'black')


# In[6]:


from sklearn.svm import SVC


# In[7]:


y = df['Virus Present']


# In[8]:


X = df.drop('Virus Present', axis=1)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[11]:


from sklearn.preprocessing import MinMaxScaler


# In[12]:


scaler = MinMaxScaler()


# In[13]:


scaled_X_train = scaler.fit_transform(X_train)


# In[14]:


scaled_X_test = scaler.transform(X_test)


# In[15]:


model = SVC(C=1000, kernel='linear')


# In[16]:


model.fit(scaled_X_train,y_train)


# In[17]:


from svm_margin_plot import plot_svm_boundary


# In[18]:


# scaled_X_train.values


# In[19]:


plt.figure(figsize=(8,6), dpi=80)

plot_svm_boundary(model=model, X=scaled_X_train, y=y_train)


# In[20]:


model = SVC(kernel='linear', C=0.05)

# C inversely projected


# In[21]:


model.fit(scaled_X_train,y_train)


# In[22]:


plt.figure(figsize=(8,6), dpi=80)

plot_svm_boundary(model=model, X=scaled_X_train, y=y_train)


# In[75]:


model = SVC(kernel='rbf', C=1, gamma = 1)

# default kernel 'rbf', gamma = 'scale'


# In[76]:


model.fit(scaled_X_train,y_train)


# In[77]:


plt.figure(figsize=(8,6), dpi=80)

plot_svm_boundary(model=model, X=scaled_X_train, y=y_train)


# In[26]:


model = SVC(kernel='sigmoid')


# In[27]:


model.fit(scaled_X_train,y_train)


# In[28]:


plt.figure(figsize=(8,6), dpi=80)

plot_svm_boundary(model=model, X=scaled_X_train, y=y_train)


# In[29]:


model = SVC(kernel='poly', C=1, degree=2)


# In[30]:


model.fit(scaled_X_train,y_train)


# In[31]:


plt.figure(figsize=(8,6), dpi=80)

plot_svm_boundary(model=model, X=scaled_X_train, y=y_train)

