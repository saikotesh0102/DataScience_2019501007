#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[14]:


path = "D:/Study/MSIT/2nd Year/DataScience_2019501007/Intro to Machine Learning/Breast Cancer Prediction - Logistic Regression/"


# In[15]:


train = path + "data.csv"
dataset = pd.read_csv(train)
print(dataset.shape)
dataset.head()


# In[16]:


X = dataset.drop('diagnosis', axis = 1)
Y = dataset["diagnosis"]
print(dataset.groupby('diagnosis').size())


# In[17]:


dataset.isnull().sum()
dataset.isna().sum()


# In[18]:


#Encoding categorical data values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[19]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[20]:


#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:


# #Using Logistic Regression Algorithm to the Training Set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, Y_train)


# In[21]:


#Fitting Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[22]:


Y_pred = classifier.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# In[24]:


Accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0])
print("The Accuracy of the model is :{}".format(Accuracy))


# In[ ]:




