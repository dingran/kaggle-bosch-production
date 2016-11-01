# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:48:51 2016

@author: dingran
"""

import util_bosch as ub
import pickle
import os
import numpy as np
import pandas as pd

with open(os.path.join(ub.code_dir, 'df_tmp_add_debug_1029.pkl'), 'r') as fpickle:
    df_tmp_add = pickle.load(fpickle)
    
col_list = list(df_tmp_add.columns)
t= '_start_date'

cols = [x for x in col_list if t in x]
                    # print cols
df_tmp_order = np.argsort(np.argsort(df_tmp_add[cols]))
df_tmp_order=pd.DataFrame(df_tmp_order)