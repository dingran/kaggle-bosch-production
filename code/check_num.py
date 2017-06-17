#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:27:32 2016

@author: dingran
"""
import pandas as pd
import util_bosch as ub

NROWS=None

df1 = pd.read_csv('../data/train_numeric.csv', nrows=NROWS)
df2 = pd.read_csv('../data/test_numeric.csv', nrows=NROWS)

df1['is_train'] = 1
df2['is_train'] = 0
df_num = pd.concat([df1, df2], ignore_index=True)
print 'Done',
print df_num.shape
del df1
del df2

TS_list = list(set(['_'.join(x.split('_')[:2]) for x in list(df_num.columns) if x.startswith('L')]))
print TS_list
print len(TS_list)

S_list = [x.split('_')[1] for x in TS_list]
print S_list
print len(S_list)

col_list = list(df_num.columns)
df_tmp_unique = df_num[['Id', 'is_train']].copy()  # to host additional columns
for ts in TS_list:
    # print ts
    cols = [x for x in col_list if ts in x]
    for c in cols:
        # print c
        existing_cols = [x for x in df_tmp_unique.columns if ts in x]
        # print existing_cols

        already_have_it = False
        for ec in existing_cols:
            if df_tmp_unique[ec].equals(df_num[c]):
                ub.log('Skipping '+c)
                already_have_it = True

        if not already_have_it:
            ub.log('Processing numeric csv, adding ' + c)
            df_tmp_unique[c] = df_num[c]
print df_tmp_unique.shape

with open('check_num.txt', 'w') as f:
    f.write('\n'.join(list(df_tmp_unique.columns)))