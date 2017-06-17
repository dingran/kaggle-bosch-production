#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 08:21:04 2016

@author: dingran
"""

import pandas as pd
import numpy as np

fname = '../data_processed/xgb_features_v0.txt'
with open(fname, 'r') as f:
    feature_list = [x.strip() for x in f.readlines()]

feature_list.append('Id')
feature_list.append('Response')

num_head = pd.read_csv('../data/train_numeric.csv', nrows=1)
date_head = pd.read_csv('../data/train_date.csv', nrows=1)
cat_head = pd.read_csv('../data/train_categorical.csv', nrows=1)

# %%
print len(num_head.columns)
num_cols = list(set(num_head.columns) & set(feature_list))
print len(num_cols)

print len(date_head.columns)
date_cols = list(set(date_head.columns) & set(feature_list))
print len(date_cols)

print len(cat_head.columns)
cat_cols = list(set(cat_head.columns) & set(feature_list))
print len(cat_cols)

# %%
NROWS = None
NROWS = 20000

df1 = pd.read_csv('../data/train_date.csv', usecols=date_cols, nrows=NROWS)
df2 = pd.read_csv('../data/test_date.csv', usecols=date_cols, nrows=NROWS)
df1['is_train'] = 1
df2['is_train'] = 0
df_date = pd.concat([df1, df2], ignore_index=True)

df1 = pd.read_csv('../data/train_categorical.csv', usecols=cat_cols, nrows=NROWS)
df2 = pd.read_csv('../data/test_categorical.csv', usecols=cat_cols, nrows=NROWS)
df1['is_train'] = 1
df2['is_train'] = 0
df_cat = pd.concat([df1, df2], ignore_index=True)

df1 = pd.read_csv('../data/train_numeric.csv', usecols=num_cols, nrows=NROWS)
if 'Response' in num_cols:
    num_cols.remove('Response')
df2 = pd.read_csv('../data/test_numeric.csv', usecols=num_cols, nrows=NROWS)
df1['is_train'] = 1
df2['is_train'] = 0
df_num = pd.concat([df1, df2], ignore_index=True)

del df1
del df2

import sys
import util_bosch as ub
from sklearn.preprocessing import LabelEncoder
#%%
na_fill_val = -1
ub.log('Processing categorical csv, fillna {}'.format(na_fill_val))
df_cat.fillna(na_fill_val, inplace=True)

ub.log('LabelEncoder running')
le = LabelEncoder()
obj_cols = df_cat.select_dtypes(include=['object']).columns
print len(obj_cols)
# print 'Obj columns: ', list(obj_cols)
counter = 0
for col in obj_cols:
    counter += 1
    print '{}/{}'.format(counter, len(obj_cols)),
    sys.stdout.flush()
    df_cat[col] = le.fit_transform(df_cat[col])

ub.log('Done processing categorical csv')





