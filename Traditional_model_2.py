#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 02:29:54 2023

@author: yewonshome
"""
import pandas as pd
import numpy as np

x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


### Scailing variables
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# change numeric data that was stored str to float
x_train['S&P 500 Weekly Close'] = x_train['S&P 500 Weekly Close'].str.replace(',', '').astype(float) 
x_test['S&P 500 Weekly Close'] = x_test['S&P 500 Weekly Close'].str.replace(',', '').astype(float) 
num_predictors = x_train.loc[:, x_train.columns != 'BSV_irregularity']
num_pred_list = num_predictors.columns.tolist()

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

# Scale each colum

preprocessor = ColumnTransformer(
    [('num', StandardScaler(), num_pred_list),
     ('cat', OrdinalEncoder(), ['BSV_irregularity'])])

# Fit and transform the column transformer on the input data
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.fit_transform(x_test)



### Decision tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train_processed, y_train)
tree_score = tree.score(x_test_processed, y_test)
tree_score


### Random Forest:
    
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(x_train_processed, y_train)
forest_score = forest.score(x_test_processed, y_test)
forest_score

predictions = forest.predict_proba(x_test_processed)



### boosted forests


### naive bayes