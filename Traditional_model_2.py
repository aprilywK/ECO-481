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

# Convert 'S&P 500 Weekly Close' column from string to float
x_train['S&P 500 Weekly Close'] = x_train['S&P 500 Weekly Close'].str.replace(',', '').astype(float) 
x_test['S&P 500 Weekly Close'] = x_test['S&P 500 Weekly Close'].str.replace(',', '').astype(float) 

# Separate numeric and categorical columns
num_predictors = x_train.loc[:, x_train.columns != 'BSV_irregularity']
num_pred_list = num_predictors.columns.tolist()

# Fill missing values with 0
x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

# Scale numeric columns and encode categorical column
preprocessor = ColumnTransformer(
    [('num', StandardScaler(), num_pred_list),
     ('cat', OrdinalEncoder(), ['BSV_irregularity'])])

# Fit and transform the column transformer on the input data
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.fit_transform(x_test)

# Change it to panda dataframe
x_train_scaled = pd.DataFrame(x_train_processed, columns=num_pred_list+['BSV_irregularity'])
x_test_scaled = pd.DataFrame(x_test_processed, columns=num_pred_list+['BSV_irregularity'])

### Decision tree

from sklearn.tree import DecisionTreeClassifier, plot_tree
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train_scaled, y_train['Sentiment'])
tree_score = tree_model.score(x_test_scaled, y_test['Sentiment'])
tree_score

# Visualize the tree
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(60,60), facecolor ='k')
plot_tree(tree_model, feature_names=x_train_scaled.columns, class_names=y_train['Sentiment'].unique(), 
          rounded=True, filled=True, fontsize=14)
plt.show()



### Random Forest:
    
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(x_train_scaled, y_train['Sentiment'])
forest_score = forest.score(x_test_scaled, y_test['Sentiment'])
forest_score

predictions = forest.predict_proba(x_test_scaled)


### boosted forests
from sklearn.ensemble import GradientBoostingClassifier

# Initialize the classifier with default hyperparameters
gbt = GradientBoostingClassifier()

# Fit the model on the training data
gbt.fit(x_train_scaled, y_train['Sentiment'])

# Evaluate the model on the test data
gbt_score = gbt.score(x_test_scaled, y_test['Sentiment'])

### naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Train the model
gnb = GaussianNB()
gnb.fit(x_train_scaled, y_train['Sentiment'].values.ravel())

# Test the model
y_pred = gnb.predict(x_test_scaled)

# Calculate accuracy
nb_score = accuracy_score(y_test['Sentiment'], y_pred)



comp_accuracy = [tree_score, forest_score, gbt_score, nb_score]



