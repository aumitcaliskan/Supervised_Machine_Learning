#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('airline_tweets.csv')


# In[4]:


df.head()


# In[6]:


sns.countplot(data=df, x='airline_sentiment')


# In[9]:


sns.countplot(data= df, x='negativereason')
plt.xticks(rotation=90);


# In[10]:


sns.countplot(data=df, x='airline', hue='airline_sentiment')


# In[11]:


data = df[['airline_sentiment','text']]


# In[12]:


data


# In[13]:


X = data['text']
y = data['airline_sentiment']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[17]:


tfidf = TfidfVectorizer(stop_words='english')


# In[19]:


tfidf.fit(X_train)


# In[20]:


X_train_tfidf = tfidf.transform(X_train)


# In[21]:


X_test_tfidf = tfidf.transform(X_test)


# In[22]:


X_train_tfidf


# In[41]:


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import plot_confusion_matrix, classification_report


# ### Naive Bayes

# In[24]:


multi_nb = MultinomialNB()


# In[25]:


multi_nb.fit(X_train_tfidf,y_train)


# In[26]:


ber_nb = BernoulliNB()


# In[27]:


ber_nb.fit(X_train_tfidf,y_train)


# ### Logistic Regression

# In[31]:


log_model = LogisticRegression(max_iter=1000)


# In[32]:


log_model.fit(X_train_tfidf,y_train)


# ### SVC

# In[34]:


rbf_svc = SVC()


# In[35]:


rbf_svc.fit(X_train_tfidf,y_train)


# In[39]:


linear_svc = LinearSVC()


# In[40]:


linear_svc.fit(X_train_tfidf,y_train)


# ### Function for models

# In[42]:


def report(model):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    plot_confusion_matrix(model, X_test_tfidf,y_test)


# In[44]:


report(multi_nb)


# In[45]:


report(ber_nb)


# In[46]:


report(log_model)


# In[47]:


report(rbf_svc)


# In[48]:


report(linear_svc)


# In[49]:


from sklearn.pipeline import Pipeline


# In[50]:


pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC())
])


# In[51]:


pipe.fit(X,y)


# In[56]:


pipe.predict(['ok flight'])


# In[ ]:




