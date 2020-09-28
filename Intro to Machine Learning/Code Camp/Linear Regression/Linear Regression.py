# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:13:18 2020

@author: Sai Koteswara Rao Ch
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mserr

path = "D:/Study/MSIT/2nd Year/DataScience_2019501007/Intro to Machine Learning/Code Camp/Linear Regression/"

train = path + "train.csv"
test = path + "test.csv"
trainx_df = pd.read_csv(train, index_col = 'Id')
# print(trainx_df.shape)

trainy_df = trainx_df['SalePrice']
# print(trainy_df.shape)

trainx_df.drop('SalePrice', axis = 1, inplace = True)
test_df = pd.read_csv(test, index_col = 'Id')
# print(test_df.shape)

sample_size = len(trainx_df)
columns_with_null_values = []
for col in trainx_df.columns:
    if trainx_df[col].isnull().sum():
        columns_with_null_values.append([col, float(trainx_df[col].isnull().sum()) / float(sample_size)])
# print(columns_with_null_values)

columns_to_drop = [x for (x,y) in columns_with_null_values if y > 0.3]
# print(columns_to_drop)

trainx_df.drop(columns_to_drop, axis = 1, inplace = True)
test_df.drop(columns_to_drop, axis = 1, inplace = True)
# print(len(trainx_df.columns))
# print(trainx_df.shape)

categorical_columns = [col for col in trainx_df.columns if trainx_df[col].dtype == object]
# categorical_columns.append('MSSubClass')
# print(len(categorical_columns))
ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
# print(len(ordinal_columns))

dummy_row = list()
for col in trainx_df.columns:
    if col in categorical_columns:
        dummy_row.append('dummy')
    else:
        dummy_row.append("")
# print(dummy_row)

new_row = pd.DataFrame([dummy_row], columns = trainx_df.columns)
trainx_df = pd.concat([trainx_df, new_row], axis = 0, ignore_index = True)
test_df = pd.concat([test_df, new_row], axis = 0, ignore_index = True)
for col in categorical_columns:
    trainx_df[col].fillna(value = 'dummy', inplace = True)
    test_df[col].fillna(value = 'dummy', inplace = True)
    
enc = OneHotEncoder(drop = 'first', sparse = False)
enc.fit(trainx_df[categorical_columns])
trainx_enc = pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
test_enc = pd.DataFrame(enc.transform(test_df[categorical_columns]))
# print(trainx_df.shape)
# print(test_df.shape)
trainx_enc.columns = enc.get_feature_names(categorical_columns)
test_enc.columns = enc.get_feature_names(categorical_columns)
trainx_df = pd.concat([trainx_df[ordinal_columns], trainx_enc], axis = 1, ignore_index = True)
test_df = pd.concat([test_df[ordinal_columns], test_enc], axis = 1, ignore_index = True)
# print(len(trainx_df.columns))

trainx_df.drop(trainx_df.tail(1).index, inplace = True)
# print(trainx_df.shape)

imputer = KNNImputer(n_neighbors = 2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled, columns = trainx_df.columns)

test_df_filled = imputer.transform(test_df)
test_df_filled = pd.DataFrame(test_df_filled, columns = test_df.columns)
test_df_filled.reset_index(drop = True, inplace = True)

print(trainx_df_filled.isnull().sum())