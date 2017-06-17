#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:38:39 2016

@author: dingran
"""
import pandas as pd
import util_bosch as ub
import seaborn as sns
import matplotlib.pyplot as plt
import os

df1 = pd.read_csv('../data_processed/df_train_date_v1_bkup.csv')
df2 = pd.read_csv('../data_processed/df_test_date_v1_bkup.csv')

tag='v1a'

df1['is_train'] = 1
df2['is_train'] = 0
df_date = pd.concat([df1, df2], ignore_index=True)
print 'Done',
print df_date.shape
del df1
del df2


response = pd.read_csv('../data/train_numeric.csv', usecols=['Response','Id'])
df_date = df_date.merge(response, on='Id', how='outer')

#%%
df_date.sort_values(by=['Id'], inplace=True)
df_date['start_date_diff'] = df_date['start_date'].diff().fillna(-12345).astype(int)



#%%
if 0:
    N_cols = len(df_date.columns)
    counter = 0
    for col in df_date.columns:
        counter += 1
        print '[plot]', counter, N_cols
        if col != 'Response':
            # g=sns.factorplot(x='Response',y=col, data=df_train_num, kind='violin')
            df = df_date[[col, 'Response']]
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
            plt.savefig(os.path.join(ub.code_dir, 'date_feature_plots', col + '.png'))
            plt.close()
#%%
#df_tmp_unique = df_date[['Id', 'is_train']].copy()  # to host additional columns
#cols = df_date.columns
#for c in cols:
#    # print c
#    existing_cols = list(df_tmp_unique.columns)
#    # print existing_cols
#
#    already_have_it = False
#    for ec in existing_cols:
#        if df_tmp_unique[ec].equals(df_date[c]):
#            ub.log('Skipping ' + c)
#
#            already_have_it = True
#
#    if not already_have_it:
#        ub.log('Processing date csv, adding ' + c)
#        df_tmp_unique[c] = df_date[c]


#%%
#import sys
#sys.exit(0)

ub.log('Done processing date csv')
df_train_date = df_date[df_date['is_train'] == 1].drop(['is_train', 'Response'], axis=1)
df_test_date = df_date[df_date['is_train'] == 0].drop(['is_train', 'Response'], axis=1)
df_train_date.to_csv('../data_processed/df_train_date_{}.csv'.format(tag), index=False)
df_test_date.to_csv('../data_processed/df_test_date_{}.csv'.format(tag), index=False)
print df_train_date.shape
print df_test_date.shape
del df_train_date
del df_test_date
