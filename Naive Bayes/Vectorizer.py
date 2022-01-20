#!/usr/bin/env python
# coding: utf-8

# In[2]:


text = ['This is a car',
       'This is a new car',
       'Absolutely different car']


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer


# In[4]:


cv = CountVectorizer()


# In[5]:


cv.get_params()


# In[6]:


sparse_matrix = cv.fit_transform(text)
sparse_matrix


# In[7]:


sparse_matrix.todense()


# In[8]:


cv.vocabulary_


# #### stop_words

# In[9]:


cv_eng = CountVectorizer(stop_words='english')


# In[10]:


cv_eng.fit_transform(text)


# In[11]:


cv_eng.vocabulary_


# ### TfidTransformer

# In[13]:


tfidf = TfidfTransformer()


# In[14]:


sparse_matrix


# In[16]:


results = tfidf.fit_transform(sparse_matrix)

# BOW to TF-IDF


# In[40]:


results.todense()


# ### TfIdVectorizer

# In[47]:


tv = TfidfVectorizer(norm='l1')

# 'l1': Sum of absolute values of vector elements is 1


# In[32]:


tv_results = tv.fit_transform(text)

# text to TF-IDF


# In[33]:


tv_results.todense()


# In[48]:


tv_2 = TfidfVectorizer()

# 'l2': Sum of squares of vector elements is 1


# In[45]:


tv_results_2 = tv_2.fit_transform(text)


# In[46]:


tv_results_2.todense()


# In[ ]:




