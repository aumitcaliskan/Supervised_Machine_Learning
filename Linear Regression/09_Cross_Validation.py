#!/usr/bin/env python
# coding: utf-8

# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake:
# a model that would just repeat the labels of the samples that it has just seen would have a perfect score
# but would fail to predict anything useful on yet-unseen data. This situation is called overfitting.

# To avoid it, it is common practice when performing a (supervised) machine learning experiment
# to hold out part of the available data as a test set X_test, y_test.

# In[9]:

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[10]:


X = np.random.randn(2000,4)
a= np.array([3,7,13,-1])
b=10
e = np.random.randn(2000,1)
y = X @ a.reshape(-1,1) + b + e


# In[11]:


df = pd.DataFrame(X)


# In[12]:


df


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)


# In[14]:


X_train.shape


# In[15]:


y_train.shape


# In[16]:


X_test.shape


# In[17]:


y_test.shape


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


model = LinearRegression()


# In[20]:


model.fit(X_train,y_train)


# In[21]:


model.score(X_train,y_train)


# In[22]:


model.score(X_test,y_test)


# In[23]:


model.coef_


# In[24]:


model.intercept_


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


from sklearn.metrics import r2_score


# In[27]:


r2_score(y_test,y_pred)


# In[28]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[29]:


mean_absolute_error(y_test,y_pred)


# In[30]:


mean_squared_error(y_test,y_pred)


# In[31]:


np.sqrt(mean_squared_error(y_test,y_pred))


# By partitioning the available data into three sets, we drastically reduce the number of samples 
# which can be used for learning the model, and the results can depend on a particular random choice 
# for the pair of (train, validation) sets.

# A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:
# 
# * A model is trained using k-1 of the folds as training data;
# * the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

# The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set), which is a major advantage in problems such as inverse inference where the number of samples is very small.

# ![](grid_search_cross_validation.png)

# The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.

# In[24]:


from sklearn.model_selection import cross_val_score


# In[25]:


scores = cross_val_score(model,X_train,y_train, cv=5)


# In[26]:


scores


# In[27]:


scores.mean()


# In[28]:


scores.std()


# The cross_validate function differs from cross_val_score in two ways:
# 
# * It allows specifying multiple metrics for evaluation.
# 
# * It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.

# In[30]:


from sklearn.model_selection import cross_validate


# In[31]:


scores = cross_validate(model,X_train,y_train,return_estimator=True)


# In[32]:


scores


# ## Cross validation iterators

# Assuming that some data is Independent and Identically Distributed (i.i.d.) is making the assumption that all samples stem from the same generative process and that the generative process is assumed to have no memory of past generated samples.

# ### K-Fold

# KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.

# In[33]:


from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.datasets import make_classification


# In[34]:


X, y = make_classification(n_samples =  100, n_features=4, n_redundant=0, n_classes=2, random_state=53)


# In[35]:


X.shape


# In[36]:


y.shape


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


model = LogisticRegression()


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


model.fit(X_train,y_train)


# In[41]:


model.score(X_train,y_train)


# In[42]:


model.score(X_test,y_test)


# In[43]:


k_fold = KFold(n_splits=4)


# In[44]:


cv = cross_validate(model,X_train,y_train,cv=k_fold,return_train_score=True)


# In[17]:


# cv = cross_validate(LogisticRegression,X_train,y_train,cv=KFold(n_splits=4),return_train_score=True)


# In[45]:


cv


# In[46]:


cv['test_score'].mean()


# In[48]:


for train, test in k_fold.split(X):
    print("%s %s" % (train, test))


# ![](sphx_glr_plot_cv_indices_006.png)

# ### Repeated K-Fold

# RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, producing different splits in each repetition.

# In[49]:


from sklearn.model_selection import RepeatedKFold


# In[50]:


repeated_k_fold = RepeatedKFold(n_splits=4, n_repeats=10)


# In[51]:


cv = cross_validate(model,X_train,y_train,cv=repeated_k_fold,return_train_score=True)


# In[52]:


cv


# In[53]:


cv['test_score'].mean()


# In[54]:


for train, test in repeated_k_fold.split(X):
    print("%s %s" % (train, test))


# ### Leave One Out (LOO)

# LeaveOneOut (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one, the test set being the sample left out. Thus, for  samples, we have  different training sets and  different tests set. This cross-validation procedure does not waste much data as only one sample is removed from the training set:

# In[55]:


from sklearn.model_selection import LeaveOneOut


# In[56]:


leave_one_out = LeaveOneOut()


# In[57]:


cv = cross_validate(model,X_train,y_train,cv=repeated_k_fold,return_train_score=True)


# In[58]:


cv['test_score'].mean()


# In[59]:


for train, test in leave_one_out.split(X):
    print("%s %s" % (train, test))


# ### Leave P Out (LPO)

# LeavePOut is very similar to LeaveOneOut as it creates all the possible training/test sets by removing  samples from the complete set. For  samples, this produces 
#  train-test pairs. Unlike LeaveOneOut and KFold, the test sets will overlap for .

# In[60]:


from sklearn.model_selection import LeavePOut


# In[61]:


leave_p_out = LeavePOut(p=2)


# In[62]:


cv = cross_validate(model, X_train,y_train,cv=leave_p_out)


# In[63]:


cv['test_score'].mean()


# In[64]:


for train, test in leave_p_out.split(X):
    print("%s %s" % (train, test))


# ### Shuffle & Split

# The ShuffleSplit iterator will generate a user defined number of independent train / test dataset splits. Samples are first shuffled and then split into a pair of train and test sets.

# In[65]:


from sklearn.model_selection import ShuffleSplit


# In[66]:


shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2)


# In[67]:


for train,test in shuffle_split.split(X):
    print('%s %s'% (train,test))


# ![](sphx_glr_plot_cv_indices_008.png)

# ## Cross-validation iterators with stratification based on class labels

# Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use stratified sampling as implemented in StratifiedKFold and StratifiedShuffleSplit to ensure that relative class frequencies is approximately preserved in each train and validation fold.

# ### Stratified k-fold

# In[68]:


from sklearn.model_selection import StratifiedKFold


# In[69]:


strafied_k_fold = StratifiedKFold(n_splits=4)


# In[70]:


for train,test in strafied_k_fold.split(X,y):
    print('train - {} | test - {}'.format(train,test))


# ![](sphx_glr_plot_cv_indices_009.png)

# ### Stratified Shuffle Split

# In[71]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[72]:


strafied_shuffle_split = StratifiedShuffleSplit(n_splits=4)


# In[73]:


for train,test in strafied_shuffle_split.split(X,y):
    print('train - {} | test - {}'.format(train,test))


# ![](sphx_glr_plot_cv_indices_012.png)

# ## Cross-validation iterators for grouped data

# Such a grouping of data is domain specific. An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient. And such data is likely to be dependent on the individual group. In our example, the patient id for each sample will be its group identifier.
# 
# In this case we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.

# ### Group k-fold

# In[74]:


from sklearn.model_selection import GroupKFold


# In[86]:


X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
X


# In[87]:


y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
y


# In[88]:


groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]


# In[89]:


group_k_fold = GroupKFold(n_splits=4)


# In[90]:


for train, test in group_k_fold.split(X, y, groups=groups):
    print("%s %s" %(train, test))


# Each subject is in a different testing fold, and the same subject is never in both testing and training.

# ![](sphx_glr_plot_cv_indices_007.png)

# ### StratifiedGroupKFold

# In[94]:


from sklearn.model_selection import StratifiedGroupKFold


# In[95]:


X = list(range(18))
X


# In[96]:


y = [1] * 6 + [0] * 12
y


# In[97]:


groups = [1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6]


# In[98]:


stratified_group_k_fold = StratifiedGroupKFold(n_splits=3)


# In[103]:


for train, test in stratified_group_k_fold.split(X,y,groups=groups):
    print("%s %s" % (train,test))


# With the current implementation full shuffle is not possible in most scenarios. When shuffle=True, the following happens:
# 
#     1. All groups a shuffled.
# 
#     2. Groups are sorted by standard deviation of classes using stable sort.
# 
#     3. Sorted groups are iterated over and assigned to folds.
# 
# That means that only groups with the same standard deviation of class distribution will be shuffled, which might be useful when each group has only a single class.
# 
# * The algorithm greedily assigns each group to one of n_splits test sets, choosing the test set that minimises the variance in class distribution across test sets. Group assignment proceeds from groups with highest to lowest variance in class frequency, i.e. large groups peaked on one or few classes are assigned first.
# 
# * This split is suboptimal in a sense that it might produce imbalanced splits even if perfect stratification is possible. If you have relatively close distribution of classes in each group, using GroupKFold is better.

# ![](sphx_glr_plot_cv_indices_005.png)

# ![](sphx_glr_plot_cv_indices_011.png)

# In[130]:


import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# In[131]:


np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4


# In[132]:


n_points = 100
X = np.random.randn(100,10)


# In[133]:


percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[i] * int(100 * perc) for i, perc in enumerate(percentiles_classes)])


# In[134]:


groups = np.hstack([[i] * 10 for i in range(10)])


# In[138]:


def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["Data\ngroup", "Data\nclass"],
        xlabel="Sample index",
    )


# In[139]:


visualize_groups(y, groups, "no groups")


# In[140]:


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for i, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [i + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [i + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [i + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


# In[141]:


fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)


# In[142]:


uneven_groups = np.sort(np.random.randint(0, 10, n_points))

cvs = [StratifiedKFold, GroupKFold, StratifiedGroupKFold]
# %%
for cv in cvs:
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_cv_indices(cv(n_splits), X, y, uneven_groups, ax, n_splits)
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    plt.show()


# %%
