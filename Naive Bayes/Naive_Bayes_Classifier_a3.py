#!/usr/bin/env python
# coding: utf-8

# ## Naive Bayes
# 
# #### Calismanin Amaclari:
# - Devamli degiskenlerde Naive bayes'in uygulama ile pekistirilmesi
# - Goruntu tasnifi

# In[1]:


from sklearn import datasets, base, naive_bayes, model_selection, svm, pipeline, preprocessing, linear_model, tree, metrics, feature_extraction, feature_selection
import scipy.stats as scs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex, Markdown
import PIL


# In[2]:


data =datasets.load_digits()


# In[3]:


display(Markdown(data['DESCR']))


# In[4]:


X, y = data['data'], data['target']


# In[5]:


fig, ax = plt.subplots(5, 10, figsize=(18, 9), sharex=True, sharey=True)
for i in range(10):
    for j in range(5):
        ax[j, i].imshow(X[y==i][j].reshape(8, 8))
fig.suptitle("Handwritten Digit Recognition Dataset Samples")
plt.tight_layout()


# ## Sorular:
# 1. Rakam tasniflendirme icin uygun Naive Bayes modelini egitip degerlendiriniz.
#     - Hangi tip dagilim en iyi calisti?
#     - Gaussian NB'ye ozel olarak, `var_smoothing` parametresini `0` olarak ayarladiginizda bir problemle karsilastiniz mi?
#         - Problemin sebebi ne olabilir?
# 2. Etiketlerin dagilimina gore; hangi olcumler model performansini degerlendirmek icin uygun olabilir?
# 3. `1` rakamlari icin recall skorunun en az %80 olmasi istenmektedir. Bunu gerceklestirmek icin hangi parametreleri kullanirsiniz?
# 4. Naive Bayes algoritmasi, bu veri seti icin uygun bir secim midir? Neden? Tartisiniz.

# In[541]:


X.shape


# In[542]:


y.shape


# In[543]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=101)


# ## BernoulliNB

# There may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be represented as binary-valued feature vectors; if handed any other kind of data, a BernoulliNB instance may binarize its input (depending on the binarize parameter).
# 
# **The decision rule for Bernoulli naive Bayes is based on function which differs from multinomial NB’s rule in that it explicitly penalizes the non-occurrence of a feature 'i' that is an indicator for class 'y', where the multinomial variant would simply ignore a non-occurring feature.**
# 
# In the case of text classification, word occurrence vectors (rather than word count vectors) may be used to train and use this classifier. BernoulliNB might perform better on some datasets, especially those with shorter documents. It is advisable to evaluate both models, if time permits.

# In[658]:


ber_model = naive_bayes.BernoulliNB(alpha=0.01,class_prior=[0.01,0.1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])


# In[659]:


ber_model.fit(X_train,y_train)


# In[660]:


preds = ber_model.predict(X_test)


# In[661]:


metrics.matthews_corrcoef(y_test,preds)


# In[662]:


metrics.jaccard_score(y_test,preds,average='macro')


# In[663]:


print(metrics.classification_report(y_test,preds))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds);


# ## MultinomialNB

# MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice). 
# 
# The smoothing priors **alpha > 0** accounts for features not present in the learning samples and prevents zero probabilities in further computations. Setting **alpha = 1** is called Laplace smoothing, while **alpha < 0** is called Lidstone smoothing.

# In[733]:


multi_model = naive_bayes.MultinomialNB(class_prior=[0.01,0.1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])


# In[734]:


multi_model.fit(X_train,y_train)


# In[735]:


preds = multi_model.predict(X_test)


# In[736]:


metrics.recall_score(y_test,preds, average='macro')


# In[737]:


metrics.matthews_corrcoef(y_test,preds)


# In[738]:


metrics.jaccard_score(y_test,preds,average='macro')


# In[739]:


print(metrics.classification_report(y_test,preds))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds);


# ## GaussianNB

# The likelihood of the features is assumed to be Gaussian

# In[620]:


gau_model = naive_bayes.GaussianNB(priors=[0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                                  var_smoothing=0)


# #### var_smooting=0

# As alpha increases, the likelihood probability drives towards uniform distribution i.e. the probability value will be 0.5 (Using higher alpha values will push the likelihood towards a value of 0.5). Most of the time, alpha = 1 is being used to resolve the problem of zero probability in the Naive Bayes algorithm.

# A Gaussian curve can serve as a "low pass" filter, allowing only the samples close to its mean to "pass." In the context of Naive Bayes, assuming a Gaussian distribution is essentially giving more weights to the samples closer to the distribution mean. This might or might not be appropriate depending if what you want to predict follows a normal distribution.
# The variable, var_smoothing, artificially adds a user-defined value to the distribution's variance. This essentially widens (or "smooths") the curve and accounts for more samples that are further away from the distribution mean.

# In[621]:


gau_model.fit(X_train,y_train)


# In[622]:


preds = gau_model.predict(X_test)


# In[623]:


metrics.recall_score(y_test,preds, average='macro')


# In[624]:


metrics.matthews_corrcoef(y_test,preds)


# In[625]:


metrics.jaccard_score(y_test,preds,average='macro')


# In[626]:


print(metrics.classification_report(y_test,preds))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds)


# ## ComplementNB

# Implements the complement naive Bayes (CNB) algorithm. CNB is an adaptation of the standard multinomial naive Bayes (MNB) algorithm that is particularly suited for imbalanced data sets. Specifically, CNB uses statistics from the complement of each class to compute the model’s weights. The inventors of CNB show empirically that the parameter estimates for CNB are more stable than those for MNB. Further, CNB regularly outperforms MNB (often by a considerable margin) on text classification tasks. 

# In[719]:


com_model = naive_bayes.ComplementNB(class_prior=[0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])


# In[720]:


com_model.fit(X_train,y_train)


# In[721]:


preds = com_model.predict(X_test)


# In[722]:


metrics.recall_score(y_test,preds, average='macro')


# In[723]:


metrics.matthews_corrcoef(y_test,preds)


# In[724]:


metrics.jaccard_score(y_test,preds,average='macro')


# In[725]:


print(metrics.classification_report(y_test,preds))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds)


# ## CategoricalNB

# Implements the categorical naive Bayes algorithm for categorically distributed data. It assumes that each feature, which is described by the index i , has its own categorical distribution.

# In[705]:


cat_model = naive_bayes.CategoricalNB()

# class_prior=[0.01,0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]


# In[706]:


cat_model.fit(X_train,y_train)


# In[707]:


preds = cat_model.predict(X_test)


# In[708]:


metrics.recall_score(y_test,preds, average='macro')


# In[709]:


metrics.matthews_corrcoef(y_test,preds)


# In[710]:


metrics.jaccard_score(y_test,preds,average='macro')


# In[711]:


print(metrics.classification_report(y_test,preds))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds)


# ### Sonuc

# CategoricalNB default verilerle en iyi skorları verdi. 
