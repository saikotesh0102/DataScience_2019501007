#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[2]:


path = "D:/Study/MSIT/2nd Year/DataScience_2019501007/Intro to Machine Learning/Graduate Admission Prediction - Linear Regression/"


# In[3]:


train = path + "Admission_Predict.csv"
df = pd.read_csv(train)
print(df.shape)
df.columns


# In[4]:


X = df.drop(['Serial No.', 'Chance of Admit '], axis = 1)
Y = df['Chance of Admit ']
X.head()


# In[5]:


df.isnull().sum()
df.isna().sum()


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
model = LinearRegression()
model.fit(X_train, Y_train)


# In[7]:


intercept = model.intercept_
coefficient = model.coef_
# print(coefficient)
print('Admit = {0:0.2f} + ({1:0.3f} * X_train)'.format(intercept, coefficient[0]))


# In[8]:


predictions = model.predict(X_test)
predictions.shape


# In[9]:


metrics_df = pd.DataFrame({'Metric' : ["Mean Absolute error", "Mean Squared error", "Root Mean Squared Error"], 
                           'Value' : [metrics.mean_absolute_error(Y_test, predictions), 
                                     metrics.mean_squared_error(Y_test, predictions),
                                     np.sqrt(metrics.mean_squared_error(Y_test, predictions))]})
metrics_df


# In[10]:


rf_model = RandomForestRegressor(n_estimators = 100,random_state = 42)
rf_model.fit(X_train,Y_train)
print('Mean absolute error for RF model: %0.4f' %metrics.mean_absolute_error(Y_test,rf_model.predict(X_test)))


# In[11]:


feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns = ['Value','Feature']) 
plt.figure(figsize = (10, 6)) 
sns.barplot(x = "Value", y="Feature", data = feature_importance.sort_values(by = "Value", ascending = False)) 


# In[ ]:




