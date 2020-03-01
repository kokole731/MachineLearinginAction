#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn import datasets
iris_data = datasets.load_iris()


Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris_data.data,
                                               iris_data.target,
                                               random_state = 1)



clf = GaussianNB()
clf.fit(Xtrain, Ytrain)

