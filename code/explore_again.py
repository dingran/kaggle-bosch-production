# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:50:34 2016

@author: dingran
"""

import pandas as pd
import util_bosch as ub
import gc
import numpy as np
import sys
import seaborn as sns
import os
import matplotlib.pyplot as plt
#import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

do_plot = True
ub.log('Starting')

N_max = 24
tag = 'v1'
testsize = 0.25

print 'N_max', N_max
print 'tag', tag
N = range(N_max)

if 0:
    ub.log('Reading date csv')
    if N_max < 24:
        dfs = []
        for i in N:
            print i,
            sys.stdout.flush()
            dfs.append(pd.read_csv('../data/train_date_part{}.csv'.format(i)))
        df1 = pd.concat(dfs, ignore_index=True)
        
        dfs = []
        for i in N:
            print i,
            sys.stdout.flush()
            dfs.append(pd.read_csv('../data/test_date_part{}.csv'.format(i)))
        df2 = pd.concat(dfs, ignore_index=True)
    else:
        df1 = pd.read_csv('../data/train_date.csv')
        df2 = pd.read_csv('../data/test_date.csv')
    
    
    df1['is_train'] = 1
    df2['is_train'] = 0
    df_date = pd.concat([df1, df2], ignore_index=True)
    print 'Done',
    print df_date.shape
    del df1
    del df2
    
    # %%
    TS_list = list(set(['_'.join(x.split('_')[:2]) for x in list(df_date.columns) if x.startswith('L')]))
    print TS_list
    print len(TS_list)
    
    S_list = [x.split('_')[1] for x in TS_list]
    print S_list
    print len(S_list)
    
    df_tmp_unique = df_date[['Id', 'is_train']].copy()  # to host additional columns
    col_list = list(df_date.columns)
    df_tmp_unique['active_ts_cnt'] = 0
    df_tmp_unique['null_col_count'] = df_date.isnull().sum(axis=1)
    df_tmp_unique['start_date'] = df_date.drop(['Id', 'is_train'], 1).min(axis=1)
    df_tmp_unique['end_date'] = df_date.drop(['Id', 'is_train'], 1).max(axis=1)
    df_tmp_unique['time_span'] = df_tmp_unique.end_date - df_tmp_unique.start_date
    
    print df_tmp_unique.shape
    
    # %%
    for ts in TS_list:
        cols = [x for x in col_list if ts in x]
        df_tmp_unique[ts + '_start_date'] = df_date[cols].min(axis=1)
        df_tmp_unique[ts + '_end_date'] = df_date[cols].max(axis=1)
    
        if df_tmp_unique[ts + '_start_date'].equals(df_tmp_unique[ts + '_end_date']):
            print 'col {} has start_date==end_date'.format(ts)
            # print df_tmp_unique.columns
            df_tmp_unique.drop([ts + '_end_date'], axis=1, inplace=True)
        else:
            print '{} has more interseting date cols'.format(ts)
            df_tmp_unique[ts + '_time_span'] = df_tmp_unique[ts + '_end_date'] - df_tmp_unique[ts + '_start_date']
            for c in cols:
                existing_cols = [x for x in df_tmp_unique.columns if ts in x]
                # print existing_cols
    
                already_have_it = False
                for ec in existing_cols:
                    if df_tmp_unique[ec].equals(df_date[c]):
                        already_have_it = True
    
                if not already_have_it:
                    ub.log('Processing date csv, adding ' + c)
                    df_tmp_unique[c] = df_date[c]
    
        df_tmp_unique[ts + '_active'] = df_tmp_unique[ts + '_start_date'].notnull().astype(int)
        df_tmp_unique['active_ts_cnt'] = df_tmp_unique['active_ts_cnt'] + df_tmp_unique[ts + '_active']
    
    df_tmp_unique['time_per_TS'] = df_tmp_unique.time_span / (1e-9 + df_tmp_unique['active_ts_cnt'])
    
    print df_tmp_unique.shape
    del df_date
    
    # %%
    col_list = list(df_tmp_unique.columns)
    ub.log('Processing date csv, adding _rank cols')
    for t in ['_start_date', '_end_date']:
        # print t
        cols = [x for x in col_list if t in x]
        # print cols
        dd = df_tmp_unique[cols]
        mask = np.isnan(dd)
        dd = np.argsort(dd)
        dd[mask] = np.nan
        dd = np.argsort(dd)
        dd[mask] = -1
        df_tmp_order = dd
        # df_tmp_order = np.argsort(np.argsort(df_tmp_unique[cols]))
        df_tmp_order = pd.DataFrame(df_tmp_order)
        df_tmp_order['Id'] = df_tmp_unique['Id']
        df_tmp_unique = pd.merge(df_tmp_unique, df_tmp_order, on='Id', suffixes=['', '_rank'])
    
    print df_tmp_unique.shape
    del df_tmp_order
    # %%
    ub.log('Adding id_diff cols')
    diff_period = 1
    dates_cols = [c for c in list(df_tmp_unique.columns) if ('start_date' in c or 'end_date' in c) and ('rank' not in c)]
    for c in dates_cols:
        # print c
        df_tmp_unique.sort_values(by=[c, 'Id'], inplace=True)
        df_tmp_unique[c + '_id_diff'] = df_tmp_unique['Id'].diff(diff_period).fillna(999999).astype(int)
        df_tmp_unique[c + '_id_diff_reverse'] = df_tmp_unique['Id'].iloc[::-1].diff().fillna(999999).astype(int)
        df_tmp_unique[c + '_id_diff_magic'] = \
            1 + 2 * (df_tmp_unique[c + '_id_diff'] > 1) + 1 * (df_tmp_unique[c + '_id_diff_reverse'] < -1)
    
        if c in ['start_date', 'end_date']:
            new_c = c + '_response_magic'
            df_tmp_unique[new_c] = 100 * df_tmp_unique[c + '_id_diff_magic'] + 10 * df_tmp_unique['L3_S32_active'] + \
                                   df_tmp_unique['L3_S33_active']
            df_tmp_unique[new_c + '_b'] = 1 * ((df_tmp_unique[new_c] == 310) | (df_tmp_unique[new_c] == 110))
    
    print df_tmp_unique.shape
    # %%
    
    na_fill_val = -9
    ub.log('Processing date csv, fillna {}'.format(na_fill_val))
    df_tmp_unique.fillna(na_fill_val, inplace=True)
    
    ub.log('Done processing date csv')
    df_train_date = df_tmp_unique[df_tmp_unique['is_train'] == 1].drop(['is_train'], axis=1)
    df_test_date = df_tmp_unique[df_tmp_unique['is_train'] == 0].drop(['is_train'], axis=1)
    df_train_date.to_csv('../data_processed/df_train_date_{}.csv'.format(tag), index=False)
    df_test_date.to_csv('../data_processed/df_test_date_{}.csv'.format(tag), index=False)
    print df_train_date.shape
    print df_test_date.shape
    del df_tmp_unique
    del df_train_date
    del df_test_date
    gc.collect()

# %%
if 0:
    ub.log('Reading categorical csv')
    
    fname = '../data_processed/xgb_features_v0.txt'
    with open(fname, 'r') as f:
        feature_list = [x.strip() for x in f.readlines()]
    feature_list.append('Id')

    cat_head = pd.read_csv('../data/train_categorical.csv', nrows=1)
    print len(cat_head.columns)
    cat_cols = list(set(cat_head.columns) & set(feature_list))
    print len(cat_cols)
    
    
    if N_max < 24:
        dfs = []
        for i in N:
            print i,
            sys.stdout.flush()
            dfs.append(pd.read_csv('../data/train_categorical_part{}.csv'.format(i), low_memory=False))
        df1 = pd.concat(dfs, ignore_index=True)
        
        dfs = []
        for i in N:
            print i,
            sys.stdout.flush()
            dfs.append(pd.read_csv('../data/test_categorical_part{}.csv'.format(i), low_memory=False))
        df2 = pd.concat(dfs, ignore_index=True)
    else:
        df1 = pd.read_csv('../data/train_categorical.csv', usecols=cat_cols)
        df2 = pd.read_csv('../data/test_categorical.csv', usecols=cat_cols)
        
    
    df1['is_train'] = 1
    df2['is_train'] = 0
    df_cat = pd.concat([df1, df2], ignore_index=True)
    print 'Done',
    print df_cat.shape
    del df1
    del df2
        
    col_list = list(df_cat.columns)
    if N_max < 24:
        df_tmp_unique = df_cat[['Id', 'is_train']].copy()  # to host additional columns
        for ts in TS_list:
            # print ts
            cols = [x for x in col_list if ts in x]
            for c in cols:
                # print c
                existing_cols = [x for x in df_tmp_unique.columns if ts in x]
                # print existing_cols
        
                already_have_it = False
                for ec in existing_cols:
                    if df_tmp_unique[ec].equals(df_cat[c]):
                        already_have_it = True
        
                if not already_have_it:
                    ub.log('Processing categorical csv, adding ' + c)
                    df_tmp_unique[c] = df_cat[c]
        print df_tmp_unique.shape
        del df_cat
    else:
        df_tmp_unique = df_cat
        del df_cat
    
    # %%
    na_fill_val = -1
    ub.log('Processing categorical csv, fillna {}'.format(na_fill_val))
    df_tmp_unique.fillna(na_fill_val, inplace=True)
    
    ub.log('LabelEncoder running')
    le = LabelEncoder()
    obj_cols = df_tmp_unique.select_dtypes(include=['object']).columns
    print len(obj_cols)
    # print 'Obj columns: ', list(obj_cols)
    counter = 0
    for col in obj_cols:
        if col == 'is_train':
            continue
        counter += 1
        print '{}/{}'.format(counter, len(obj_cols)),
        sys.stdout.flush()
        df_tmp_unique[col] = le.fit_transform(df_tmp_unique[col])
    
    # %%
    ub.log('Done processing categorical csv')
    print df_tmp_unique.shape
    df_train_cat = df_tmp_unique[df_tmp_unique['is_train'] == 1].drop(['is_train'], axis=1)
    df_test_cat = df_tmp_unique[df_tmp_unique['is_train'] == 0].drop(['is_train'], axis=1)
    df_train_cat.to_csv('../data_processed/df_train_cat_{}.csv'.format(tag), index=False)
    df_test_cat.to_csv('../data_processed/df_test_cat_{}.csv'.format(tag), index=False)
    print df_train_cat.shape
    print df_test_cat.shape
    del df_tmp_unique
    gc.collect()

    
#%%
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
sys.exit(0)

na_fill_val = -19
df_train_date = pd.read_csv('../data_processed/df_train_date_{}.csv'.format(tag))

df_train = pd.merge(df_train_date, df_train_cat, on='Id', copy=False)
del df_train_date
del df_train_cat
gc.collect()

print 'reading train numeric'
if N_max < 24:
    dfs = []
    for i in N:
        print i,
        sys.stdout.flush()
        dfs.append(pd.read_csv('../data/train_numeric_part{}.csv'.format(i)))
    df_train_num = pd.concat(dfs, ignore_index=True)
else:
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

# %%

if do_plot and False:
    N_cols = len(df_train.columns)
    counter = 0
    for col in df_train.columns:
        counter += 1
        print '[plot]', counter, N_cols
        if col != 'Response':
            # g=sns.factorplot(x='Response',y=col, data=df_train_num, kind='violin')
            df = df_train[[col, 'Response']]
            # plt.figure()
            fig, ax = plt.subplots(2)
            # sns.violinplot(x='Response', y=col, kind='violin', data=df)
            # sns.kdeplot(df[df['Response']==0][col], ax=ax, cumulative=True)
            # sns.kdeplot(df[df['Response']==1][col], ax=ax, cumulative=True)
            x = df[df['Response'] == 0][col].dropna().values
            ax[0].hist(x, cumulative=True, normed=True, alpha=0.3)
            ax[1].hist(x, normed=True, alpha=0.3)
            x = df[df['Response'] == 1][col].dropna().values
            ax[0].hist(x, cumulative=True, normed=True, alpha=0.3)
            ax[1].hist(x, normed=True, alpha=0.3)
            # response0_null_pct = df[df['Response'] == 0][col].notnull().sum()*1.0/df[df['Response'] == 0][col].shape[0]
            # response1_null_pct = df[df['Response'] == 1][col].notnull().sum()*1.0/df[df['Response'] == 1][col].shape[0]
            # plt.title('{}, null pct: response0: {}, response1 {}'.format(col, response0_null_pct, response1_null_pct))
            plt.title(col)
            plt.legend(['Response 0', 'Response 1'])
            plt.savefig(os.path.join(ub.code_dir, 'feature_plots', col + '.png'))
            plt.close()

# %%
sys.exit(0)
import xgboost as xgb
keep_ID = True

columns_to_drop = ['Id', 'Response']
if keep_ID:
    columns_to_drop = ['Response']

y_total_df = df_train['Response']

shuffle_col = df_train[['Id']].copy()
shuffle_col['Id'] = np.random.rand(len(shuffle_col))

train_idx = shuffle_col[shuffle_col['Id'] > testsize].index
val_idx = shuffle_col[shuffle_col['Id'] <= testsize].index
ub.log('Done shuffling...')
print 'len of train_idx', len(train_idx)
print 'len of val_idx', len(val_idx)
y_train = y_total_df.loc[train_idx].values
y_val = y_total_df.loc[val_idx].values

df_train.drop(columns_to_drop, axis=1, inplace=True)
feature_names = list(df_train.columns)
with open('../data_processed/xgb_features_{}.txt'.format(tag), 'w') as ff:
    ff.write('\n'.join(feature_names))

print df_train.shape
xgtrain = xgb.DMatrix(df_train[train_idx], label=y_train, feature_names=feature_names)
xgtrain.save_binary('../data_processed/xgtrain_{}.buffer'.format(tag))
xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names)
xgval.save_binary('../data_processed/xgval_{}.buffer'.format(tag))

with open('feature_select_1102.txt', 'r') as ff:
    selected_features = [fe.strip() for fe in ff.readlines()]

df_train = df_train[selected_features]
feature_names_sub = list(df_train.columns)
with open('../data_processed/xgb_features_sub_{}.txt'.format(tag), 'w') as ff:
    ff.write('\n'.join(feature_names_sub))

print df_train.shape
xgtrain = xgb.DMatrix(df_train[train_idx], label=y_train, feature_names=feature_names_sub)
xgtrain.save_binary('../data_processed/xgtrain_sub_{}.buffer'.format(tag))
xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names_sub)
xgval.save_binary('../data_processed/xgval_sub_{}.buffer'.format(tag))

selected_features_tiny = selected_features[:50]
df_train = df_train[selected_features_tiny]
feature_names_tiny = list(df_train.columns)
with open('../data_processed/xgb_features_tiny_{}.txt'.format(tag), 'w') as ff:
    ff.write('\n'.join(feature_names_tiny))

print df_train.shape
xgtrain = xgb.DMatrix(df_train[train_idx], label=y_train, feature_names=feature_names_tiny)
xgtrain.save_binary('../data_processed/xgtrain_tiny_{}.buffer'.format(tag))
xgval = xgb.DMatrix(df_train[val_idx], label=y_val, feature_names=feature_names_tiny)
xgval.save_binary('../data_processed/xgval_tiny_{}.buffer'.format(tag))

del df_train
del xgtrain

df_test_date = pd.read_csv('../data_processed/df_test_date_{}.csv'.format(tag))
df_test = pd.merge(df_test_cat, df_test_date, on='Id', copy=False)
del df_test_cat
del df_test_date
gc.collect()

print 'reading test numeric'
if N_max < 24:
    dfs = []
    for i in N:
        print i,
        sys.stdout.flush()
        dfs.append(pd.read_csv('../data/test_numeric_part{}.csv'.format(i)))
    df_test_num = pd.concat(dfs, ignore_index=True)
else:
    df_train_num = pd.read_csv('../data/test_numeric.csv', usecols=num_cols)
    
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

if not keep_ID:
    columns_to_drop = ['Id']
    df_test.drop(columns_to_drop, axis=1, inplace=True)
# feature_names = list(df_train.columns)
print df_test.shape
xgtest = xgb.DMatrix(df_test, feature_names=feature_names)
xgtest.save_binary('../data_processed/xgtest_{}.buffer'.format(tag))

df_test = df_test[selected_features]
feature_names_sub = list(df_test.columns)
print df_test.shape
xgtest = xgb.DMatrix(df_test, feature_names=feature_names_sub)
xgtest.save_binary('../data_processed/xgtest_sub_{}.buffer'.format(tag))

df_test = df_test[selected_features_tiny]
feature_names_tiny = list(df_test.columns)
print df_test.shape
xgtest = xgb.DMatrix(df_test, feature_names=feature_names_tiny)
xgtest.save_binary('../data_processed/xgtest_tiny_{}.buffer'.format(tag))

del df_test
del xgtest
