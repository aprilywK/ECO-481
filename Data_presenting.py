#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:42:29 2023

@author: yewonshome
"""

### Merge Two datasets to get one dataset on financial data
import pandas as pd

# read the two Parquet files as DataFrames
df1 = pd.read_parquet('train.parquet')
df2 = pd.read_parquet('test.parquet')

# concatenate the two DataFrames
merged_df = pd.concat([df1, df2], ignore_index=True)

types = pd.DataFrame(merged_df.dtypes, columns=['Type'])
types.index.name = 'Column'

print(types.T)

print(merged_df.dtypes)
print(merged_df.info())

# Quantitative predictors
quant_pred = merged_df.drop(columns = ["summary_detail", "title", "summary_detail_with_title"])

summary1 = quant_pred.describe()
summary2 = quant_pred.describe().loc[["min", "max"]] # only range

import matplotlib.pyplot as plt

# pull up a list of all the column name
columns = list(quant_pred)

# "o" stands for circles, other letters can give other shapes

for col in columns:
    plt.plot(merged_df[col], merged_df["mpg"], "o")
    plt.xlabel(col)
    plt.ylabel("mpg")
    plt.show()