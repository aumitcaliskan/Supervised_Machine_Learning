#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate


# In[2]:


df = pd.read_csv('gene_expression.csv')


# In[3]:


df.head()


# In[4]:


sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue= 'Cancer Present', alpha=0.5, style='Cancer Present')


# In[5]:


len(df)


# In[6]:


plt.figure(figsize=(8,8))
sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue= 'Cancer Present', style='Cancer Present')

plt.xlim(2,6)
plt.ylim(4,8)


# In[7]:


sns.pairplot(data=df, hue='Cancer Present')


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


X = df.drop('Cancer Present', axis=1)


# In[11]:


y = df['Cancer Present']


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


scaler = StandardScaler()


# In[14]:


scaled_X_train = scaler.fit_transform(X_train)


# In[15]:


scaled_X_test = scaler.transform(X_test)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


model = KNeighborsClassifier(n_neighbors=1)


# In[18]:


model.fit(scaled_X_train, y_train)


# In[19]:


y_pred = model.predict(scaled_X_test)


# In[20]:


from sklearn.metrics import confusion_matrix, classification_report


# In[21]:


confusion_matrix(y_test, y_pred)


# In[22]:


len(y_test)


# In[23]:


print(classification_report(y_test,y_pred))


# In[24]:


df['Cancer Present'].value_counts()

# balanced class


# ## n_neighbors=1 optimal? 

# ### Elbow Method

# In[25]:


from sklearn.metrics import accuracy_score


# In[26]:


error = 1 - accuracy_score(y_test,y_pred)
error


# In[27]:


test_error_rates = []

for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train)
    
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1-accuracy_score(y_test,y_pred_test)
    
    test_error_rates.append(test_error)
    


# In[28]:


test_error_rates


# In[29]:


plt.figure(figsize=(8,10))
plt.plot(range(1,30), test_error_rates)
plt.ylabel('Error Rate')
plt.xlabel('K Neighbors')


# ## GridSearch CV

# In[77]:


from sklearn.model_selection import GridSearchCV


# In[79]:


scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[80]:


knn = KNeighborsClassifier()


# In[81]:


knn.get_params().keys()


# In[82]:


k = list(range(1,31))


# In[83]:


param_grid = {'algorithm':['brute', 'kd_tree', 'ball_tree'], 
              'metric':['minkowski','euclidean','manhattan','chebyshev'],
              'n_neighbors':k,
             'weights':['uniform','distance']}


# In[84]:


knn_grid = GridSearchCV(knn, param_grid, return_train_score=True)


# In[85]:


knn_grid.fit(scaled_X_train, y_train)


# In[106]:


knn_grid.score(scaled_X_train, y_train)


# In[107]:


knn_grid.score(scaled_X_test, y_test)


# In[108]:


# pd.DataFrame(knn_grid.cv_results_).head()


# In[87]:


y_pred = knn_grid.predict(scaled_X_test)


# In[88]:


print(classification_report(y_test,y_pred))


# **weights** default=’uniform’
# 
# ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
# 
# ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
# 
# **algorithm** default=’auto’
# 
# ‘brute’ will use a brute-force search. for N samples in D dimensions, distance = D * N**2
# 
#     * Brute-force search or exhaustive search is a very general problem-solving technique that consists of systematically enumerating all possible candidates for the solution and checking whether each candidate satisfies the problem's statement.
#         1. Compute all the distances between the query point and reference points.
#         2. Sort the computed distances.
#         3. Select the k reference points with the smallest distances.
#         4. Classification vote by k nearest objects.
#         5. Repeat steps (1 to 4) for all query points.
#         
# ‘kd_tree’ will use KDTree distance = D * N * log(N)
# 
#      * To address the computational inefficiencies of the brute-force approach. The basic idea is that if point A is very distant from point B, and point B is very close to point C, then we know that points A and C are very distant, without having to explicitly calculate their distance. In this way, the computational cost of a nearest neighbors search can be reduced to D*N*log(N) or better. This is a significant improvement over brute-force for large 
#      * Though the KD tree approach is very fast for low-dimensional D < 20
#      * sklearn.neighbors.KDTree(X, leaf_size=40, metric='minkowski', **kwargs)¶
# 
# ‘ball_tree’ will use BallTree
# 
#     * To address the inefficiencies of KD Trees in higher dimensions. Where KD trees partition data along Cartesian axes, ball trees partition data in a series of nesting hyper-spheres. This makes tree construction more costly than that of the KD tree, but results in a data structure which can be very efficient on highly structured data, even in very high dimensions.
#     * sklearn.neighbors.BallTree(X, leaf_size=40, metric='minkowski', **kwargs)¶
# 
# ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
# 
# **leaf_size** default=30
# 
# Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. 
# 
# **metric** default=’minkowski’
# 
# “euclidean” EuclideanDistance sqrt(sum((x - y)^2))
# 
# “manhattan” ManhattanDistance sum(|x - y|)
# 
# “chebyshev” ChebyshevDistance max(|x - y|)
# 
# “minkowski” MinkowskiDistance p, w sum(w * |x - y|^p)^(1/p)
# 
# “wminkowski” WMinkowskiDistance p, w sum(|w * (x - y)|^p)^(1/p)
# 
# “seuclidean” SEuclideanDistance V sqrt(sum((x - y)^2 / V))
# 
# “mahalanobis” MahalanobisDistance V or VI sqrt((x - y)' V^-1 (x - y))
# 
# **metric_params** default=None
# 
# **p** default=2
# 
# Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
# 
# **n_neighbors** default=5

# ## Pipeline

# *Note: If your parameter grid is going inside a PipeLine, your parameter name needs to be specified in the following manner:**
# 
# * chosen_string_name + **two** underscores + parameter key name
# * model_name + __ + parameter name
# * knn_model + __ + n_neighbors
# * knn_model__n_neighbors
# 
# [StackOverflow on this](https://stackoverflow.com/questions/41899132/invalid-parameter-for-sklearn-estimator-pipeline)
# 
# The reason we have to do this is because it let's scikit-learn know what operation in the pipeline these parameters are related to (otherwise it might think n_neighbors was a parameter in the scaler).

# In[41]:


operations = [('scaler',scaler), ('knn',knn)]  


# In[42]:


from sklearn.pipeline import Pipeline 


# In[43]:


pipe = Pipeline(operations) 


# In[44]:


k_values = list(range(1,20))


# In[45]:


param_grid = {'knn__n_neighbors':k_values} 


# In[46]:


full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')


# In[47]:


full_cv_classifier.fit(X_train,y_train)


# In[48]:


full_cv_classifier.best_estimator_.get_params()


# In[49]:


full_pred = full_cv_classifier.predict(X_test)


# In[50]:


print(classification_report(y_test,full_pred))


# In[51]:


new_patient = [[3.8,6.4]]


# In[52]:


full_cv_classifier.predict(new_patient)


# In[53]:


full_cv_classifier.predict_proba(new_patient)


# ## Data with Noise

# In[54]:


from sklearn.datasets import make_blobs


# In[55]:


X ,y  = make_blobs(n_samples=100,n_features=2, centers=2, cluster_std=4, random_state=42 )


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=5, weights='distance', p=3, algorithm='kd_tree', metric='manhattan' )
model.fit(scaled_X_train, y_train)


# In[57]:


k_degerler = list(range(1,40))
train= []
test = []

for k in k_degerler:
    for a 
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=3, algorithm='brute', metric='minkowski' )
    cv = cross_validate(estimator=model, X=scaled_X_train, y=y_train, cv=10, return_train_score=True, n_jobs=-1)
    train.append(cv["train_score"].mean())
    test.append(cv["test_score"].mean())


# In[58]:


plt.figure(figsize=(12,8))
plt.plot(k_degerler, train, label="train")
plt.plot(k_degerler, test, label="test")
plt.legend()
plt.xlabel('k')
plt.ylabel('score')


# ## KNN pros and cons

# **Advantages of KNN**
# 
# 1. No Training Period: KNN is called Lazy Learner (Instance based learning). It does not learn anything in the training period. It does not derive any discriminative function from the training data. In other words, there is no training period for it. It stores the training dataset and learns from it only at the time of making real time predictions. This makes the KNN algorithm much faster than other algorithms that require training e.g. SVM, Linear Regression etc.
# 
# 2. Since the KNN algorithm requires no training before making predictions, new data can be added seamlessly which will not impact the accuracy of the algorithm.
# 
# 3. KNN is very easy to implement. There are only two parameters required to implement KNN i.e. the value of K and the distance function (e.g. Euclidean or Manhattan etc.)

# **Disadvantages of KNN**
# 
# 1. Does not work well with large dataset: In large datasets, the cost of calculating the distance between the new point and each existing points is huge which degrades the performance of the algorithm. It's not just the CPU that takes a hit with k Nearest Neighbor, RAM also gets occupied when this little monster is working. kNN stores all its values in the RAM
# 
# 2. Does not work well with high dimensions: The KNN algorithm doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate the distance in each dimension.
# 
# 3. Need feature scaling: We need to do feature scaling (standardization and normalization) before applying KNN algorithm to any dataset. If we don't do so, KNN may generate wrong predictions.
# 
# 4. Sensitive to noisy data, missing values and outliers: KNN is sensitive to noise in the dataset. We need to manually impute missing values and remove outliers.

# In[ ]:




