# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:31:56 2016

@author: dingran
"""
import pandas as pd
import os
import util_bosch as ub

df = pd.read_csv(os.path.join(ub.output_dir, 'feature_importance_xgb_accumu_list_df.csv'))

target_features = list(set(df.sort_values(by=['fscore'])['feature'].values[0:250]))

f_test = 'load_data_record_col_names_Test_2016-10-25 20:31:38.txt'
f_train = 'load_data_record_col_names_Train_2016-10-25 20:30:30.txt'
with open(os.path.join(ub.output_dir, f_train), 'r') as f:
    col_list_all = [x.strip() for x in f.readlines()]

print len(target_features)
print len(col_list_all)

actual_features = []
for feature in target_features:
    if 'id_diff' in feature:
        print feature
        actual_features.append(feature.replace('_id_diff', '').replace('_reverse', ''))
    else:
        actual_features.append(feature)


print actual_features
print len(actual_features)
print len(set(actual_features))

actual_features = list(set(actual_features))

for x in actual_features:
    if x not in col_list_all:
        print x