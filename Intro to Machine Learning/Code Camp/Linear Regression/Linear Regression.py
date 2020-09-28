#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mserr


# In[2]:


path = "D:/Study/MSIT/2nd Year/DataScience_2019501007/Intro to Machine Learning/Code Camp/Linear Regression/"


# In[3]:


train = path + "train.csv"
test = path + "test.csv"
trainx_df = pd.read_csv(train, index_col = 'Id')
print(trainx_df.shape)


# In[4]:


trainy_df = trainx_df['SalePrice']
print(trainy_df.shape)


# In[5]:


trainx_df.drop('SalePrice', axis = 1, inplace = True)
test_df = pd.read_csv(test, index_col = 'Id')
print(test_df.shape)


# In[6]:


sample_size = len(trainx_df)
columns_with_null_values = []
for col in trainx_df.columns:
    if trainx_df[col].isnull().sum():
        columns_with_null_values.append([col, float(trainx_df[col].isnull().sum()) / float(sample_size)])
print(columns_with_null_values)


# In[7]:


columns_to_drop = [x for (x,y) in columns_with_null_values if y > 0.3]
print(columns_to_drop)


# In[8]:


trainx_df.drop(columns_to_drop, axis = 1, inplace = True)
test_df.drop(columns_to_drop, axis = 1, inplace = True)
print(len(trainx_df.columns))
print(trainx_df.shape)


# In[9]:


categorical_columns = [col for col in trainx_df.columns if trainx_df[col].dtype == object]
# categorical_columns.append('MSSubClass')
# print(len(categorical_columns))
ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
# print(len(ordinal_columns))


# In[10]:


dummy_row = list()
for col in trainx_df.columns:
    if col in categorical_columns:
        dummy_row.append('dummy')
    else:
        dummy_row.append("")
print(dummy_row)


# In[11]:


new_row = pd.DataFrame([dummy_row], columns = trainx_df.columns)
trainx_df = pd.concat([trainx_df, new_row], axis = 0, ignore_index = True)
test_df = pd.concat([test_df, new_row], axis = 0, ignore_index = True)
for col in categorical_columns:
    trainx_df[col].fillna(value = 'dummy', inplace = True)
    test_df[col].fillna(value = 'dummy', inplace = True)


# In[12]:


enc = OneHotEncoder(drop = 'first', sparse = False)
enc.fit(trainx_df[categorical_columns])
trainx_enc = pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
test_enc = pd.DataFrame(enc.transform(test_df[categorical_columns]))
print(trainx_df.shape)
print(test_df.shape)
trainx_enc.columns = enc.get_feature_names(categorical_columns)
test_enc.columns = enc.get_feature_names(categorical_columns)
trainx_df = pd.concat([trainx_df[ordinal_columns], trainx_enc], axis = 1, ignore_index = True)
test_df = pd.concat([test_df[ordinal_columns], test_enc], axis = 1, ignore_index = True)
print(len(trainx_df.columns))


# In[13]:


trainx_df.drop(trainx_df.tail(1).index, inplace = True)
test_df.drop(test_df.tail(1).index, inplace = True)
print(trainx_df.shape)


# In[14]:


imputer = KNNImputer(n_neighbors = 2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled, columns = trainx_df.columns)
print(trainx_df_filled.isnull().sum())


# In[15]:


test_df_filled = imputer.transform(test_df)
test_df_filled = pd.DataFrame(test_df_filled, columns = test_df.columns)
test_df_filled.reset_index(drop = True, inplace = True)


# In[16]:


scaler = preprocessing.StandardScaler().fit(trainx_df)
trainx_df = scaler.transform(trainx_df_filled)
test_df = scaler.transform(test_df_filled)
print(trainx_df.shape)
print(test_df.shape)


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(trainx_df, trainy_df.values.ravel(), test_size = 0.3, random_state = 42)


# In[24]:


LRModel = LinearRegression().fit(X_train, Y_train)


score_train = []
score_test = []
mse_train = []
mse_test = []
alpha = []
alpha_start = 2
alpha_end = 146
jumps = 10

for sigma in np.linspace(alpha_start, alpha_end, jumps):
    alpha.append(sigma)
    # Ridge_model=getRidgeRegressionModel(X_train, y_train,reg_par=sigma)
    Ridge_model = Ridge(alpha = sigma,tol = 0.01).fit(X_train, Y_train)
    score_train.append(round(Ridge_model.score(X_train, Y_train), 10))
    score_test.append(round(Ridge_model.score(X_test, Y_test), 10))
    mse_train.append(round(mserr(Y_train,Ridge_model.predict(X_train)), 4))
    mse_test.append(round(mserr(Y_test,Ridge_model.predict(X_test)), 4))
print(alpha,'\n', score_train, '\n', score_test,'\n', mse_train, '\n', mse_test) 

Ridge_model = Ridge(alpha = 146,tol = 0.01).fit(X_train, Y_train)
testpred = pd.DataFrame(Ridge_model.predict(test_df))
testpred.to_csv("test_pred.csv")

plt.figure(1)
plt.plot(alpha, score_train, 'g--',label = "train_score")
plt.plot(alpha, score_test, 'r-o',label = "test_score")
plt.xlabel = 'Alpha'
plt.legend()
plt.figure(2)
plt.plot(alpha, mse_train, 'y--',label = "train_mse")
plt.plot(alpha, mse_test, 'c-o',label = "test_mse")
plt.xlabel = 'Alpha'
plt.legend()
plt.show()


# In[21]:


def predictTestx(Model, testx_df):
    testpred = pd.DataFrame(Model.predict(testx_df))
    testpred.to_csv("test_pred.csv")


# In[ ]:




