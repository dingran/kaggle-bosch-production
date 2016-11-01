# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:18:46 2016

@author: dingran
"""

import pandas as pd

STATIONS = ['S32', 'S33', 'S34']
train_date_part = pd.read_csv('../data/train_date.csv', nrows=10000)
date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
date_cols = date_cols[date_cols['station'].isin(STATIONS)]
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
print(date_cols)
train_date = pd.read_csv('../data/train_date.csv', usecols=['Id'] + date_cols)
print(train_date.columns)
train_date.columns = ['Id'] + STATIONS