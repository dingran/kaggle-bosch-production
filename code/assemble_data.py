#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:41:52 2016

@author: dingran
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import util_bosch as ub
import gc

tag = 'v1'
keep_ID = True
na_fill_val = -19
testsize = 0.2

num_head = pd.read_csv('../data/train_numeric.csv', nrows=1)
if 1:
    fname = '../code/check_num.txt'
    with open(fname, 'r') as f:
        feature_list = [x.strip() for x in f.readlines()]

    feature_list.append('Id')
    feature_list.append('Response')

    print len(num_head.columns)
    num_cols = list(set(num_head.columns) & set(feature_list))
    print len(num_cols)
else:
    num_cols = list(num_head.columns)

# %%
df_train_date = pd.read_csv('../data_processed/df_train_date_{}a.csv'.format(tag))
df_train_cat = pd.read_csv('../data_processed/df_train_cat_{}.csv'.format(tag))

df_train = pd.merge(df_train_date, df_train_cat, on='Id', copy=False)
del df_train_date
del df_train_cat
gc.collect()

print 'reading train numeric'
df_train_num = pd.read_csv('../data/train_numeric.csv', usecols=num_cols)

print df_train_num.shape
df_train = df_train.merge(df_train_num, on='Id', copy=False)
del df_train_num
gc.collect()

cols_full_flag = df_train.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))

df_train.fillna(na_fill_val, inplace=True)

cols_full_flag = df_train.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))

obj_cols = df_train.select_dtypes(include=['object']).columns
print 'Object columns: {}'.format(len(obj_cols))
df_train.to_csv('../data_processed/df_train_overall_{}.csv'.format(tag), index=False)

#
# columns_to_drop = ['Id', 'Response']
# if keep_ID:
#    columns_to_drop = ['Response']
#
# y_total_df = df_train['Response']
#
# shuffle_col = df_train[['Id']].copy()
# shuffle_col['Id'] = np.random.rand(len(shuffle_col))
#
# train_idx = shuffle_col[shuffle_col['Id'] > testsize].index
# val_idx = shuffle_col[shuffle_col['Id'] <= testsize].index
# ub.log('Done shuffling...')
# print 'len of train_idx', len(train_idx)
# print 'len of val_idx', len(val_idx)
# y_train = y_total_df.loc[train_idx].values
# y_val = y_total_df.loc[val_idx].values
#
# df_train.drop(columns_to_drop, axis=1, inplace=True)
# feature_names = list(df_train.columns)
# with open('../data_processed/xgb_features_{}.txt'.format(tag), 'w') as ff:
#    ff.write('\n'.join(feature_names))
#
# print df_train.shape
# xgtrain = xgb.DMatrix(df_train.loc[train_idx], label=y_train, feature_names=feature_names)
# xgtrain.save_binary('../data_processed/xgtrain_{}.buffer'.format(tag))
# xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names)
# xgval.save_binary('../data_processed/xgval_{}.buffer'.format(tag))
#
# with open('feature_select_1102.txt', 'r') as ff:
#    selected_features = [fe.strip() for fe in ff.readlines()]
#
# df_train = df_train[selected_features]
# feature_names_sub = list(df_train.columns)
# with open('../data_processed/xgb_features_sub_{}.txt'.format(tag), 'w') as ff:
#    ff.write('\n'.join(feature_names_sub))
#
# print df_train.shape
# xgtrain = xgb.DMatrix(df_train[train_idx], label=y_train, feature_names=feature_names_sub)
# xgtrain.save_binary('../data_processed/xgtrain_sub_{}.buffer'.format(tag))
# xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names_sub)
# xgval.save_binary('../data_processed/xgval_sub_{}.buffer'.format(tag))
#
# selected_features_tiny = selected_features[:50]
# df_train = df_train[selected_features_tiny]
# feature_names_tiny = list(df_train.columns)
# with open('../data_processed/xgb_features_tiny_{}.txt'.format(tag), 'w') as ff:
#    ff.write('\n'.join(feature_names_tiny))
#
# print df_train.shape
# xgtrain = xgb.DMatrix(df_train[train_idx], label=y_train, feature_names=feature_names_tiny)
# xgtrain.save_binary('../data_processed/xgtrain_tiny_{}.buffer'.format(tag))
# xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names_tiny)
# xgval.save_binary('../data_processed/xgval_tiny_{}.buffer'.format(tag))

del df_train
gc.collect()
# del xgtrain

# %%
df_test_date = pd.read_csv('../data_processed/df_test_date_{}a.csv'.format(tag))
df_test_cat = pd.read_csv('../data_processed/df_test_cat_{}.csv'.format(tag))

df_test = pd.merge(df_test_cat, df_test_date, on='Id', copy=False)
del df_test_cat
del df_test_date
gc.collect()

print 'reading test numeric'
if 'Response' in num_cols:
    print 'remove Response in num_cols'
    num_cols.remove('Response')
df_test_num = pd.read_csv('../data/test_numeric.csv', usecols=num_cols)

print df_test_num.shape
df_test = df_test.merge(df_test_num, on='Id', copy=False)
del df_test_num
gc.collect()

cols_full_flag = df_test.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))

df_test.fillna(na_fill_val, inplace=True)

cols_full_flag = df_test.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))

obj_cols = df_test.select_dtypes(include=['object']).columns
print 'Object columns: {}'.format(len(obj_cols))
df_test.to_csv('../data_processed/df_test_overall_{}.csv'.format(tag), index=False)

# if not keep_ID:
#    columns_to_drop = ['Id']
#    df_test.drop(columns_to_drop, axis=1, inplace=True)
## feature_names = list(df_train.columns)
# print df_test.shape
# xgtest = xgb.DMatrix(df_test, feature_names=feature_names)
# xgtest.save_binary('../data_processed/xgtest_{}.buffer'.format(tag))
#
# df_test = df_test[selected_features]
# feature_names_sub = list(df_test.columns)
# print df_test.shape
# xgtest = xgb.DMatrix(df_test, feature_names=feature_names_sub)
# xgtest.save_binary('../data_processed/xgtest_sub_{}.buffer'.format(tag))
#
# df_test = df_test[selected_features_tiny]
# feature_names_tiny = list(df_test.columns)
# print df_test.shape
# xgtest = xgb.DMatrix(df_test, feature_names=feature_names_tiny)
# xgtest.save_binary('../data_processed/xgtest_tiny_{}.buffer'.format(tag))

del df_test
# del xgtest
