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


df.info()


# Attribute Information:
# 
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# In[5]:


sns.countplot(data=df, x='class' )


# In[10]:


feat_uni = df.describe().T.reset_index().sort_values('unique')


# In[14]:


plt.figure(figsize=(14,6), dpi=200)
sns.barplot(data=feat_uni, x='index', y='unique')
plt.xticks(rotation=90);


# In[15]:


X = df.drop('class', axis=1)


# In[17]:


X = pd.get_dummies(X, drop_first=True)


# In[18]:


X


# In[19]:


y = df['class']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[22]:


from sklearn.ensemble import AdaBoostClassifier


# In[23]:


model = AdaBoostClassifier(n_estimators=1)


# In[24]:


model.fit(X_train, y_train)


# In[40]:


from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score


# In[26]:


predictions = model.predict(X_test)


# In[27]:


predictions


# In[28]:


print(classification_report(y_test, predictions))


# In[31]:


model.feature_importances_

# n_estimators = 1


# In[32]:


model.feature_importances_.argmax()


# In[34]:


X.columns[22]


# In[36]:


sns.countplot(data=df, x='odor', hue='class')


# In[37]:


len(X.columns)


# In[41]:


error_rates = []

for n in range(1,96):
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    err = 1 - accuracy_score(y_test, preds)
    error_rates.append(err)
    


# In[47]:


plt.plot(range(1,96),error_rates)


# In[48]:


model


# In[53]:


feats = pd.DataFrame(index=X.columns, data=model.feature_importances_,columns=['Importance'])


# In[56]:


feats_imp = feats[feats['Importance']>0]


# In[57]:


feats_imp


# In[63]:


plt.figure(figsize=(12,6))
sns.barplot(data=feats_imp.sort_values('Importance'), x=feats_imp.index, y='Importance')
plt.xticks(rotation=90);


# In[ ]:




