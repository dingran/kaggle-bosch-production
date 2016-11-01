# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:50:34 2016

@author: dingran
"""

import pandas as pd
import util_bosch as ub

N=range(9)

dfs =[]
for i in N:
    dfs.append(pd.read_csv('../data/train_numeric_part{}.csv'.format(i)))
df_train_num = pd.concat(dfs, ignore_index=True)
resp = df_train_num[['Id', 'Response']].copy()

dfs = []
for i in N:
    dfs.append(pd.read_csv('../data/train_date_part{}.csv'.format(i)))
df_train_date = pd.concat(dfs, ignore_index=True)
df_train_date = pd.merge(df_train_date, resp, on='Id')
#df_train_cat = pd.read_csv('../data/train_cat_part19.csv')

#%%

import seaborn as sns
import os

N_cols = len(df_train_date.columns)
counter=0
for col in df_train_date.columns:
    counter+=1
    print counter, N_cols    
    if col !='Response':
        g=sns.factorplot(x='Response',y=col, data=df_train_date, kind='violin')
        g.savefig(os.path.join(ub.code_dir, 'feature_plots', col+'.png'))
        

N_cols = len(df_train_num.columns)
counter=0
for col in df_train_num.columns:
    counter+=1
    print '[2]', counter, N_cols    
    if col !='Response':
        g=sns.factorplot(x='Response',y=col, data=df_train_num, kind='violin')
        g.savefig(os.path.join(ub.code_dir, 'feature_plots', col+'.png'))
        g.plt.close()
