#!/usr/bin/env python
# coding: utf-8

# In[14]:


with open('One.txt') as mytext:
    a = mytext.read()


# In[15]:


a


# In[16]:


print(a)


# In[19]:


with open('One.txt') as mytext:
    b = mytext.readlines()


# In[20]:


b


# In[21]:


a.lower().split()


# In[22]:


with open('Two.txt') as mytext:
    c = mytext.read()


# In[24]:


c.lower().split()


# In[25]:


with open('One.txt') as mytext:
    words_one = mytext.read().lower().split()
    uni_words_one = set(words_one)


# In[26]:


uni_words_one


# In[27]:


with open('Two.txt') as mytext:
    words_two = mytext.read().lower().split()
    uni_words_two = set(words_two)


# In[28]:


uni_words_two


# In[30]:


all_uni_words = set()
all_uni_words.update(uni_words_one)


# In[31]:


all_uni_words.update(uni_words_two)


# In[32]:


all_uni_words


# In[33]:


full_vocab = dict()
i = 0

for word in all_uni_words:
    full_vocab[word] = i
    i = i+1


# In[34]:


full_vocab


# In[35]:


one_freq = [0]*len(full_vocab)
two_freq = [0]*len(full_vocab)
all_words = ['']*len(full_vocab)


# In[36]:


one_freq


# In[37]:


all_words


# In[39]:


with open('One.txt') as f:
    one_text = f.read().lower().split()


# In[40]:


one_text


# In[42]:


for word in one_text:
    word_ind = full_vocab[word]
    one_freq[word_ind] += 1


# In[43]:


one_freq


# In[44]:


with open('Two.txt') as f:
    two_text = f.read().lower().split()


# In[45]:


for word in two_text:
    word_ind = full_vocab[word]
    two_freq[word_ind] += 1


# In[46]:


two_freq


# In[47]:


all_words


# In[48]:


for word in full_vocab:
    word_ind = full_vocab[word]
    all_words[word_ind] = word


# In[49]:


all_words


# In[51]:


import pandas as pd
bow = pd.DataFrame(data=[one_freq,two_freq], columns=all_words)
bow


# In[ ]:




