# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:04:12 2016

@author: dingran
"""
import json
import util_bosch as ub
import os
import sys
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
import datetime
import matplotlib.pyplot as plt
import gc
import operator
import seaborn as sns
import glob
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker, Timer
import argparse

# from memory_profiler import profile

pd.set_option('display.max_colwidth', -1)
N_DEBUG_LINES = 10000


def get_params(bases_core=None):
    seed = 1019
    if bases_core is None:
        bases_core = 0.05
    # https://github.com/dmlc/xgboost/blob/master/doc/param_tuning.md
    params = {
        # 1- General Parameters
        # 'booster': "gbtree",  # booster [default=gbtree]
        'silent': True,  # silent [default=0]
        # 'nthread' : 8 , #nthread [default to maximum number of threads available if not set]

        # 2A-Parameters for Tree Booster
        'eta': 0.02,  # 0.023, # eta [default=0.3] range: [0,1]
        # 'gamma':0 ,#gamma [default=0] range: [0,inf]
        'max_depth': 6,  # max_depth [default=6] range: [1,inf]
        'min_child_weight': 3,  # default=1]range: [0,inf]
        # 'max_delta_step':0, #max_delta_step [default=0] range: [0,inf]
        'subsample': 0.7,  # 0.83, #subsample [default=1]range: (0,1]
        'colsample_bytree': 0.77,  # 0.77, #colsample_bytree [default=1]range: (0,1]
        # 'lambda': 1,  #lambda [default=1]
        # 'alpha':0.0001, #alpha [default=0]


        # 2B- Parameters for Linear Booster
        # 'lambda': 0,  #lambda [default=0]
        # 'alpha':0, #alpha [default=0]
        # 'lambda_bias':0, #default 0

        # 3- earning Task Parameters
        'objective': 'binary:logistic',  # objective [ default=reg:linear ]
        'base_score': bases_core,  # base_score [ default=0.5 ]
        # 'eval_metric': 'logloss',  # eval_metric [ default according to objective ]
        'seed': seed  # seed [ default=0 ]

    }
    return params


def load_data(load_test=False,
              original_cols_only=False,
              N_start=None, N_read=5, N_split=24,
              shuffle=False,
              feature_list_file=None,
              load_date_csv=True,
              load_numerical_csv=True,
              load_categorical_csv=True):
    assert load_categorical_csv or load_numerical_csv or load_date_csv

    if load_test:
        ub.log('load test files')
    else:
        ub.log('load train files')

    print 'N_start: {}'.format(N_start),
    if N_start is None:
        print ', i.e. random start'
    else:
        print ''
    print 'N_read: {}'.format(N_read)
    print 'N_splits: {}'.format(N_split)
    print 'shuffle: {}'.format(shuffle)
    print 'original_cols_only: {}'.format(original_cols_only)
    print 'feature_list_file: {}'.format(feature_list_file)

    if shuffle:
        N_list = range(N_split)
        random.shuffle(N_list)
        file_ids = N_list[:N_read]
    else:
        if N_start is None:
            N_start = random.randint(0, N_split - N_read)
        N_list = range(N_start, N_read + N_start)
        file_ids = N_list

    print file_ids,

    basefname_list = ub.train_files
    if load_test:
        basefname_list = ub.test_files

    print basefname_list

    original_date_cols = []
    original_num_cols = []
    original_cat_cols = []

    # widgets = ['reading', ': ', Percentage(), ' ', Bar(), ' ', ETA()]
    # pbar = ProgressBar(widgets=widgets, maxval=N_read).start()
    counter = 0
    df_output = None
    for id in file_ids:
        # print id
        counter += 1

        df_chunk = None
        for basefname in basefname_list:

            basefname_part_template = ub.data_fname_to_partial_data_fname_template(basefname)
            input_fname_template = os.path.join(ub.data_dir, basefname_part_template)
            input_fname = input_fname_template.format(id)

            ub.log('reading {}'.format(input_fname))

            if 'date' in basefname:
                if not load_date_csv:
                    print 'skip date table...'
                    continue
                else:
                    if os.path.exists(input_fname + '.pkl'):
                        df_tmp = pickle.load(input_fname + '.pkl')
                    else:
                        df_tmp = pd.read_csv(input_fname, low_memory=False)
                    if not original_date_cols:
                        original_date_cols = list(df_tmp.columns)
                    df_tmp_add = df_tmp[['Id']].copy()  # to host additional columns
                    if not original_cols_only:
                        df_tmp_add['null_col_count'] = df_tmp.isnull().sum(axis=1)
                        df_tmp_add['start_date'] = df_tmp.drop('Id', 1).min(axis=1)
                        df_tmp_add['end_date'] = df_tmp.drop('Id', 1).max(axis=1)
                        df_tmp_add['time_span'] = df_tmp_add.end_date - df_tmp_add.start_date
                        TS_list = list(
                            set(['_'.join(x.split('_')[:2]) for x in list(df_tmp.columns) if x.startswith('L')]))
                        # print TS_list
                        col_list = list(df_tmp.columns)
                        df_tmp_add['active_ts_cnt'] = 0
                        for ts in TS_list:
                            cols = [x for x in col_list if ts in x]
                            df_tmp_add[ts + '_start_date'] = df_tmp[cols].min(axis=1)
                            df_tmp_add[ts + '_end_date'] = df_tmp[cols].max(axis=1)
                            df_tmp_add[ts + '_time_span'] = df_tmp_add[ts + '_end_date'] - df_tmp_add[
                                ts + '_start_date']
                            # add TS active flag
                            df_tmp_add[ts + '_active'] = df_tmp_add[ts + '_start_date'].notnull().astype(int)
                            df_tmp_add['active_ts_cnt'] = df_tmp_add['active_ts_cnt'] + df_tmp_add[ts + '_active']

                        # added TS ordering
                        # ub.log('Adding TS ordering columns')
                        col_list = list(df_tmp_add.columns)
                        # with open(os.path.join(ub.code_dir, 'df_tmp_add_debug_1029.pkl'), 'w') as fpickle:
                        #     pickle.dump(df_tmp_add, fpickle)
                        for t in ['_start_date', '_end_date', '_time_span']:
                            # print t
                            cols = [x for x in col_list if t in x]
                            # print cols
                            df_tmp_order = np.argsort(np.argsort(df_tmp_add[cols]))
                            df_tmp_order = pd.DataFrame(df_tmp_order)
                            df_tmp_order['Id'] = df_tmp_add['Id']
                            df_tmp_add = pd.merge(df_tmp_add, df_tmp_order, on='Id', suffixes=['', '_rank'])

            elif 'numeric' in basefname:
                if not load_numerical_csv:
                    if not load_test:
                        print 'skip numerical table... but will still read Response column'
                        df_tmp = pd.read_csv(input_fname, usecols=['Id', 'Response'])
                        df_tmp_add = df_tmp[['Id']].copy()  # to host additional columns
                    else:
                        print 'skip numerical table...'
                        continue
                else:
                    if os.path.exists(input_fname + '.pkl'):
                        df_tmp = pickle.load(input_fname + '.pkl')
                    else:
                        df_tmp = pd.read_csv(input_fname, low_memory=False)

                    if not original_num_cols:
                        original_num_cols = list(df_tmp.columns)
                    df_tmp_add = df_tmp[['Id']].copy()  # to host additional columns

                    df_tmp_add['num_null_col_count'] = df_tmp.isnull().sum(axis=1)
                    df_tmp_add['num_active_ts_cnt'] = 0
                    TS_list = list(set(['_'.join(x.split('_')[:2]) for x in list(df_tmp.columns) if x.startswith('L')]))
                    col_list = list(df_tmp.columns)
                    for ts in TS_list:
                        cols = [x for x in col_list if ts in x]
                        # print cols
                        df_tmp_add[ts + '_num_active'] = (df_tmp[cols].notnull().sum(axis=1) / len(cols)).astype(int)
                        df_tmp_add['num_active_ts_cnt'] = df_tmp_add['num_active_ts_cnt'] + df_tmp_add[
                            ts + '_num_active']
            elif 'categorical' in basefname:
                if not load_categorical_csv:
                    print 'skip categorical table...'
                    continue
                else:
                    if os.path.exists(input_fname + '.pkl'):
                        df_tmp = pickle.load(input_fname + '.pkl')
                    else:
                        df_tmp = pd.read_csv(input_fname, low_memory=False)

                    if not original_cat_cols:
                        original_cat_cols = list(df_tmp.columns)
                    df_tmp_add = df_tmp[['Id']].copy()  # to host additional columns

                    df_tmp_add['cat_null_col_count'] = df_tmp.isnull().sum(axis=1)
                    df_tmp_add['cat_active_ts_cnt'] = 0
                    TS_list = list(set(['_'.join(x.split('_')[:2]) for x in list(df_tmp.columns) if x.startswith('L')]))
                    col_list = list(df_tmp.columns)
                    for ts in TS_list:
                        cols = [x for x in col_list if ts in x]
                        # print cols
                        df_tmp_add[ts + '_cat_active'] = (df_tmp[cols].notnull().sum(axis=1) / len(cols)).astype(int)
                        df_tmp_add['cat_active_ts_cnt'] = df_tmp_add['cat_active_ts_cnt'] + df_tmp_add[
                            ts + '_cat_active']
            else:
                assert 0

            print 'df_tmp shape:', df_tmp.shape
            selected_features = []
            if feature_list_file is not None:
                # cols_selected = ub.cols[keyword]

                feature_list_file_full_path = os.path.join(ub.code_dir, os.path.basename(feature_list_file))
                ub.log('Using feature list" {}'.format(feature_list_file_full_path), 'highlight')

                with open(feature_list_file_full_path, 'r') as f_feature:
                    selected_features = [x.strip() for x in f_feature.readlines()]

                if 'Id' not in selected_features:
                    selected_features.append('Id')

                if 'Response' not in selected_features:
                    selected_features.append('Response')

                # ub.log('Down selecting features with N_features={}'.format(N_features))
                # if feature_print_flag == 0:
                #     # N_features = 500
                #     ub.log('Down selecting features with N_features={}'.format(N_features))
                #     df_feature = pd.read_csv(os.path.join(ub.output_dir, 'feature_importance_xgb_accumu_list_df.csv'))
                #     target_features = list(set(df_feature.sort_values(by=['fscore'])['feature'].values[:N_features]))
                #
                #     f_test = 'load_data_record_col_names_Test_2016-10-25 20:31:38.txt'
                #     f_train = 'load_data_record_col_names_Train_2016-10-25 20:30:30.txt'
                #     with open(os.path.join(ub.output_dir, f_train), 'r') as f:
                #         col_list_all = [x.strip() for x in f.readlines()]
                #     # print len(target_features)
                #     # print len(col_list_all)
                #
                #     actual_features = []
                #     for feature in target_features:
                #         if 'id_diff' in feature:
                #             # print feature
                #             actual_features.append(
                #                 feature.replace('_id_diff', '').replace('_reverse', '').replace('_magic', ''))
                #             # these are created after loading in data
                #         else:
                #             actual_features.append(feature)
                #     # print actual_features
                #     # print len(actual_features)
                #     # print len(set(actual_features))
                #     actual_features = list(set(actual_features))
                #
                #     selected_features = []
                #     for x in actual_features:
                #         if x in col_list_all:
                #             selected_features.append(x)
                #         else:
                #             ub.log('Found feature: {}, not in col_list_all, removing it'.format(x), 'error')
                #
                #     selected_features.append('Id')
                #     if not load_test:
                #         selected_features.append('Response')
                #
                #     ub.log('Features selected ({}):'.format(len(selected_features)), 'highlight')
                #     # print '\n'.join(selected_features)
                #     # print selected_features
                #     feature_print_flag += 1

                # print df_tmp.columns
                df_tmp = df_tmp[list(set(df_tmp.columns) & set(selected_features))]
                print 'df_tmp shape (after selection):', df_tmp.shape
                # print df_tmp.shape
                # print df_tmp.columns

            print 'df_tmp_add shape (merged):', df_tmp_add.shape

            if len(set(df_tmp.columns) & set(df_tmp_add.columns)) > 1:
                print set(df_tmp.columns) & set(df_tmp_add.columns)
                assert 0

            df_tmp = df_tmp.merge(df_tmp_add, on='Id', copy=False)
            print 'df_tmp shape (merged):', df_tmp.shape
            if selected_features:
                df_tmp = df_tmp[list(set(df_tmp.columns) & set(selected_features))]
                print 'df_tmp shape (merged, after selection):', df_tmp.shape
                # print df_tmp.shape
                # print df_tmp.columns

            if df_chunk is None:
                df_chunk = df_tmp
            else:
                df_chunk = df_chunk.merge(df_tmp, on='Id', copy=False)

            print 'df_chunk shape (merged):', df_chunk.shape
            gc.collect()

        if not original_cols_only:
            if (not selected_features) or (('time_per_TS_num' in selected_features) \
                                                   and ('time_per_TS_cat' in selected_features) \
                                                   and ('time_span' in selected_features) \
                                                   and ('num_active_ts_cnt' in selected_features) \
                                                   and ('cat_active_ts_cnt' in selected_features)):
                if 'num_active_ts_cnt' in df_chunk.columns:
                    df_chunk['time_per_TS_num'] = df_chunk.time_span / (1e-9 + df_chunk['num_active_ts_cnt'])
                if 'cat_active_ts_cnt' in df_chunk.columns:
                    df_chunk['time_per_TS_cat'] = df_chunk.time_span / (1e-9 + df_chunk['cat_active_ts_cnt'])
                if 'active_ts_cnt' in df_chunk.columns:
                    df_chunk['time_per_TS'] = df_chunk.time_span / (1e-9 + df_chunk['active_ts_cnt'])
        # print 'df_chunk shape: {}'.format(df_chunk.shape)

        if df_output is None:
            df_output = df_chunk
        else:
            df_output = pd.concat([df_output, df_chunk], ignore_index=True)

        # pbar.update(counter)
        gc.collect()

    # pbar.finish()
    print 'df_output shape: {}'.format(df_output.shape)
    # dates_cols = [x for x in list(df_output.columns) if 'start_date' in x or 'end_date' in x]
    # df_output[dates_cols].head(n=1000).to_csv(os.path.join(ub.data_dir, 'df_output_debug.csv'))
    datetime_str2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    col_list_fname = os.path.join(ub.output_dir, 'load_data_record_col_names_Train_{}.txt'.format(datetime_str2))
    if load_test:
        col_list_fname = os.path.join(ub.output_dir, 'load_data_record_col_names_Test_{}.txt'.format(datetime_str2))
    with open(col_list_fname, 'w') as fp:
        fp.write('\n'.join(list(df_output.columns)))

    return df_output, N_start, list(set(original_num_cols + original_cat_cols + original_date_cols))


def split_data(df_to_split, output_fname_template=None):
    # output_fname_template exampe: os.path.join(ub.data_dir, 'df_train_preprocessed_part{}.csv')
    ub.log('splitting {}'.format(output_fname_template))

    N_split = ub.N_split

    ub.log('outputting {}'.format(output_fname_template))
    n_rows = int(df_to_split.shape[0] / N_split)

    widgets = ['splitting', ': ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=N_split).start()
    for i in range(N_split):
        output_fname = output_fname_template.format(i)
        if 0:
            if i < N_split - 1:
                print 'writng rows:', i * n_rows, (i + 1) * n_rows - 1
            else:
                print 'writng rows:', i * n_rows, df_to_split.shape[0]

        df_to_split[i * n_rows:(i + 1) * n_rows].to_csv(output_fname, index=False)
        pbar.update(i)
    pbar.finish()


# datetime_str1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# fname_mem_profile = os.path.join(ub.output_dir, 'memory_profiles/memory_profile_run_{}.log'.format(datetime_str1))
# f_mem = open(fname_mem_profile, 'w')
#
# @profile(stream=f_mem)
def main(run_info_fname=None,
         compile_data=False,
         train_model=False,
         make_submission=False,
         N_start=None,
         N_files_train=1,
         N_files_test=1,
         original_cols_only=False,
         disable_id_diff_cols=False,

         feature_list_file=None,
         analyze_feature_importance=False,
         cv=False,
         # if True running cross validation if False, run single model training session and importance analysis
         early_stop_rounds=10,
         N_rounds=1000,
         testsize=0.3,
         xgb_params=None,

         skip_date_csv=False,
         skip_num_csv=False,
         skip_cat_csv=False
         ):
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if compile_data:
        run_info = dict()
        N_splits = ub.N_split
        if N_files_train > N_splits:
            N_files_train = N_splits
        if N_files_test > N_splits:
            N_files_test = N_splits

        # if analyze_feature_importance and (feature_list_file is not None):
        #     assert 0

        run_info['compile_data'] = compile_data
        run_info['N_splits'] = N_splits
        run_info['N_start'] = N_start
        run_info['N_files_train'] = N_files_train
        run_info['N_files_test'] = N_files_test
        run_info['original_cols_only'] = original_cols_only
        run_info['disable_id_diff_cols'] = disable_id_diff_cols
        run_info['features_list_file'] = feature_list_file
        run_info['skip_date_csv'] = skip_date_csv
        run_info['skip_num_csv'] = skip_num_csv
        run_info['skip_cat_csv'] = skip_cat_csv

        df_train, n_start, orig_cols = load_data(load_test=False, N_start=N_start, N_read=N_files_train,
                                                 N_split=N_splits,
                                                 original_cols_only=original_cols_only,
                                                 feature_list_file=feature_list_file,
                                                 load_categorical_csv=(not skip_cat_csv),
                                                 load_date_csv=(not skip_date_csv),
                                                 load_numerical_csv=(not skip_num_csv))
        df_test, _1, _2 = load_data(load_test=True, N_start=n_start, N_read=N_files_test, N_split=N_splits,
                                    original_cols_only=original_cols_only,
                                    feature_list_file=feature_list_file,
                                    load_categorical_csv=(not skip_cat_csv),
                                    load_date_csv=(not skip_date_csv),
                                    load_numerical_csv=(not skip_num_csv))

        if not disable_id_diff_cols:
            diff_period = 1
            ub.log('generating id diff columns based on various dates columns: diff_period = {}'.format(diff_period))
            dates_cols = [x for x in list(df_train.columns) if
                          ('start_date' in x or 'end_date' in x) and ('rank' not in x)]

            # print dates_cols

            df_datesort = pd.concat([df_train[['Id'] + dates_cols], df_test[['Id'] + dates_cols]],
                                    ignore_index=True)
            gc.collect()

            for c in dates_cols:
                # print c
                df_datesort.sort_values(by=[c, 'Id'], inplace=True)
                df_datesort[c + '_id_diff'] = df_datesort['Id'].diff(diff_period).fillna(999999).astype(int)
                df_datesort[c + '_id_diff_reverse'] = df_datesort['Id'].iloc[::-1].diff().fillna(999999).astype(int)
                df_datesort[c + '_id_diff_magic'] = \
                    1 + 2 * (df_datesort[c + '_id_diff'] > 1) + 1 * (df_datesort[c + '_id_diff_reverse'] < -1)

                df_datesort.drop([c], axis=1, inplace=True)

            df_datesort.head(n=N_DEBUG_LINES).to_csv(os.path.join(ub.data_dir, 'df_datesort_debug.csv'), index=False)

            gc.collect()
            df_train = df_train.merge(df_datesort, on='Id')

            gc.collect()
            df_test = df_test.merge(df_datesort, on='Id')

        df_test['Response'] = 0
        df_train.head(n=N_DEBUG_LINES).to_csv(os.path.join(ub.data_dir, 'df_train_debug.csv'), index=False)
        o = set(orig_cols)
        c = set(df_train.columns)
        c.difference_update(o)
        new_cols = list(c)
        df_train[new_cols].head(n=N_DEBUG_LINES).to_csv(os.path.join(ub.data_dir, 'df_train_debug_new_cols_only.csv'),
                                                        index=False)

        df_test.head(n=N_DEBUG_LINES).to_csv(os.path.join(ub.data_dir, 'df_test_debug.csv'), index=False)
        o = set(orig_cols)
        c = set(df_test.columns)
        c.difference_update(o)
        new_cols = list(c)
        df_test[new_cols].head(n=N_DEBUG_LINES).to_csv(os.path.join(ub.data_dir, 'df_test_debug_new_cols_only.csv'),
                                                       index=False)

        print df_train.shape
        print df_test.shape
        gc.collect()

        # if N_files_train == N_splits:
        #     split_data(df_train,
        #                output_fname_template=os.path.join(ub.processed_data_dir, 'df_train_preprocessed_part{}.csv'))
        # if N_files_test == N_splits:
        #     split_data(df_test,
        #                output_fname_template=os.path.join(ub.processed_data_dir, 'df_test_preprocessed_part{}.csv'))

        fillna = True
        run_info['fillna'] = fillna
        if fillna:
            ub.log('Filling na...')
            for df in [df_train, df_test]:
                cols_full_flag = df.isnull().any()
                non_full_cols = list(cols_full_flag[cols_full_flag].index)
                print 'Non-full columns: {}'.format(len(non_full_cols))
                # print non_full_cols

                if 1:
                    df.fillna(-999999, inplace=True)
                else:
                    # print df.PersonalField7.unique()
                    for c in non_full_cols:
                        if len(df[c].unique()) > 2:
                            most_frequent_items = df[c].value_counts().idxmax()
                            print c, most_frequent_items
                            df[c].fillna(value=most_frequent_items, inplace=True)
                        else:  # if it is only a pair of value [somthing, nan] then fill in "missing"
                            df[c].fillna(value='missing', inplace=True)
                            print c, df[c].unique()

                cols_full_flag = df.isnull().any()
                non_full_cols = list(cols_full_flag[cols_full_flag].index)
                print 'Non-full columns: {}'.format(len(non_full_cols))

                le = LabelEncoder()
                obj_cols = df.select_dtypes(include=['object']).columns
                # print 'Obj columns: ', list(obj_cols)
                for col in obj_cols:
                    df[col] = le.fit_transform(df[col])

            df_train.head(n=1000).to_csv(os.path.join(ub.data_dir, 'df_train_cleanup_debug.csv'), index=False)
            df_test.head(n=1000).to_csv(os.path.join(ub.data_dir, 'df_train_cleanup_debug.csv'), index=False)

        ub.log('Dropping Id and Response columns...')
        columns_to_drop = ['Id', 'Response']
        shuffle_col = df_train[['Id']].copy()
        shuffle_col['Id'] = np.random.rand(len(shuffle_col))

        y_total_df = df_train['Response']
        y_total = df_train['Response'].values
        df_train.drop(columns_to_drop, axis=1, inplace=True)
        df_test.drop(columns_to_drop, axis=1, inplace=True)

        print df_train.shape
        print df_test.shape
        prior = np.sum(y_total) / (1. * len(y_total))
        print 'prior: {}'.format(prior)
        run_info['prior'] = prior

        gc.collect()

        feature_imp_fname_template = os.path.join(ub.output_dir, 'feature_importance_xgb_{}')
        run_info['feature_imp_fname_template'] = feature_imp_fname_template
        top_features_fname = feature_imp_fname_template.format('accumu_list.txt')
        run_info['top_features_fname'] = top_features_fname

        # if feature_down_select:
        #     ub.log('Feature down selected based on {}...'.format(top_features_fname))
        #     #todo may need to set a maxN for the number of features to use
        #
        #     with open(top_features_fname, 'r') as tf:
        #         selected_cols = [x.strip() for x in tf.readlines()]
        #     df_train = df_train[selected_cols]
        #     df_test = df_test[selected_cols]
        #     print df_train.shape
        #     print df_test.shape
        #     print df_train.columns

        feature_names = list(df_train.columns)

        postfix_train = '{}_{}of{}'.format(datetime_str, N_files_train, N_splits)
        postfix_test = '{}_{}of{}'.format(datetime_str, N_files_test, N_splits)

        run_info['postfix_train'] = postfix_train
        run_info['postfix_test'] = postfix_test

        run_info['testsize'] = testsize

        train_test_split_method = 1

        ub.log('Train/val split using testsize={}, split_method={}'.format(testsize, train_test_split_method))
        if train_test_split_method == 1:
            train_idx = shuffle_col[shuffle_col['Id'] > testsize].index
            val_idx = shuffle_col[shuffle_col['Id'] <= testsize].index
            ub.log('Done shuffling...')
            print 'len of train_idx', len(train_idx)
            print 'len of val_idx', len(val_idx)
            y_train = y_total_df.loc[train_idx].values
            y_val = y_total_df.loc[val_idx].values

            xgtrain = xgb.DMatrix(df_train.loc[train_idx].values, y_train, feature_names=feature_names)
            ub.log('Assembled xgtrain')
            xgval = xgb.DMatrix(df_train.loc[val_idx].values, y_val, feature_names=feature_names)
            ub.log('Assembled xgval')
            del df_train
            ub.log('Deleted df_train')
            gc.collect()
        else:
            x_train, x_val, y_train, y_val = train_test_split(df_train.values, y_total, test_size=testsize)
            ub.log('Done shuffling...')
            print x_train.shape
            print x_val.shape
            del df_train
            gc.collect()
            ub.log('Deleted df_train')

            xgtrain = xgb.DMatrix(x_train, y_train, feature_names=feature_names)
            ub.log('Assembled xgtrain')
            xgval = xgb.DMatrix(x_val, y_val, feature_names=feature_names)
            ub.log('Assembled xgval')
            del x_train
            del x_val
            gc.collect()

        fname_xgtrain = os.path.join(ub.processed_data_dir, 'xgtrain_{}.buffer'.format(postfix_train))
        xgtrain.save_binary(fname_xgtrain)
        ub.log('Saved {}'.format(fname_xgtrain))

        fname_xgval = os.path.join(ub.processed_data_dir, 'xgval_{}.buffer'.format(postfix_train))
        xgval.save_binary(fname_xgval)
        ub.log('Saved {}'.format(fname_xgval))

        xgtest = xgb.DMatrix(df_test.values, feature_names=feature_names)
        ub.log('Assembled xgtest')
        fname_xgtest = os.path.join(ub.processed_data_dir, 'xgtest_{}.buffer'.format(postfix_test))
        xgtest.save_binary(fname_xgtest)
        ub.log('Saved {}'.format(fname_xgtest))

        del df_test
        gc.collect()
        ub.log('Deleted df_test')

        print 'train and val set sizes'
        print xgtrain.num_row(), xgtrain.num_col()
        print xgval.num_row(), xgval.num_col()
        run_info['xgtrain_nrows'] = xgtrain.num_row()
        run_info['xgval_nrows'] = xgval.num_row()

        run_info['fname_xgtrain'] = fname_xgtrain
        run_info['fname_xgval'] = fname_xgval
        run_info['fname_xgtest'] = fname_xgtest

        fname_ytrain = os.path.join(ub.processed_data_dir, 'ytrain_{}.npy'.format(postfix_train))
        fname_yval = os.path.join(ub.processed_data_dir, 'yval_{}.npy'.format(postfix_train))

        np.save(fname_ytrain, y_train)
        ub.log('Saved ' + fname_ytrain)

        np.save(fname_yval, y_val)
        ub.log('Saved ' + fname_yval)

        run_info['fname_ytrain'] = fname_ytrain
        run_info['fname_yval'] = fname_yval

    if train_model:
        assert compile_data or (run_info_fname is not None)

        run_info['cv'] = cv
        run_info['analyze_feature_importance'] = analyze_feature_importance
        run_info['early_stop_rounds'] = early_stop_rounds

        if not compile_data:
            ub.log('(train_model) Loading run info from {} ...'.format(run_info_fname))
            with open(run_info_fname, 'r') as fp:
                run_info = eval(fp.read())
            print json.dumps(run_info, indent=2)

            run_info_fname = run_info_fname.replace('.txt', '_{}.txt'.format(datetime_str))

            logged_home_dir = None
            if ub.home_dir not in run_info['fname_xgtrain']:
                for i in ub.possible_home_dirs:
                    if i in run_info['fname_xgtrain']:
                        logged_home_dir = i

                for k in ['fname_xgtrain', 'fname_xgval', 'fname_ytrain', 'fname_yval']:
                    run_info[k] = run_info[k].replace(logged_home_dir, ub.home_dir)

                if analyze_feature_importance:
                    for k in ['feature_imp_fname_template', 'top_feature_fname']:
                        run_info[k] = run_info[k].replace(logged_home_dir, ub.home_dir)

            ub.log('Loading xgtrain data {} ...'.format(run_info['fname_xgtrain']))
            xgtrain = xgb.DMatrix(run_info['fname_xgtrain'])

            ub.log('Loading xgval data {} ...'.format(run_info['fname_xgval']))
            xgval = xgb.DMatrix(run_info['fname_xgval'])

            ub.log('Loading ytrain data {} ...'.format(run_info['fname_ytrain']))
            y_train = np.load(run_info['fname_ytrain'])

            ub.log('Loading yval data {} ...'.format(run_info['fname_yval']))
            y_val = np.load(run_info['fname_yval'])

        prior = run_info['prior']
        postfix_train = run_info['postfix_train']

        if xgb_params is None:
            xgb_params = get_params(bases_core=prior)

        xgb_params['base_score'] = prior  # n_positive / n_total
        # xgb_params['scale_pos_weight'] = (1.0 - prior) / prior
        run_info['xgb_params'] = xgb_params
        ub.log('Get xgb_params')
        print xgb_params

        xgb_num_rounds = N_rounds
        run_info['xgb_num_rounds'] = xgb_num_rounds
        print 'xgb_num_rounds', xgb_num_rounds
        if cv:
            ub.log('Running cross validation...')
            eval_hist = xgb.cv(xgb_params, xgtrain, num_boost_round=xgb_num_rounds,
                               early_stopping_rounds=early_stop_rounds,
                               feval=ub.mcc_eval, maximize=True,
                               verbose_eval=1, show_stdv=True, nfold=3, seed=0, stratified=True)
            print eval_hist
            eval_hist_fname = os.path.join(ub.output_dir, 'cv_eval_history_{}.csv'.format(postfix_train))
            if not compile_data:
                eval_hist_fname = eval_hist_fname.replace('.csv', '_{}.csv'.format(datetime_str))

            run_info['eval_hist_fname'] = eval_hist_fname
            eval_hist.to_csv(eval_hist_fname)

            run_info['cv_score_test'] = eval_hist['test-MCC-mean'].max()
            run_info['cv_score_train'] = eval_hist['train-MCC-mean'].max()

        if 1:
            ub.log('Running training...')
            watchlist = [(xgtrain, 'train'), (xgval, 'eval')]
            model = xgb.train(xgb_params, xgtrain, num_boost_round=xgb_num_rounds,
                              early_stopping_rounds=early_stop_rounds,
                              feval=ub.mcc_eval, maximize=True,
                              evals=watchlist, verbose_eval=True)

            model_fname = os.path.join(ub.output_dir, 'xbg_{}.model'.format(postfix_train))
            if not compile_data:
                model_fname = model_fname.replace('.model', '_{}.model'.format(datetime_str))
            ub.log('Saving model: {}...'.format(model_fname))
            model.save_model(model_fname)
            model.dump_model(model_fname + '.raw.txt')
            run_info['model_fname'] = model_fname

            ntree_limit = model.best_iteration + 1

            ub.log('Predictions on xgtrain...', 'highlight')
            predictions = model.predict(xgtrain, ntree_limit=ntree_limit)

            best_proba, best_mcc, y_pred = ub.eval_mcc(y_train, predictions, True)
            mcc_official = matthews_corrcoef(y_train, y_pred)
            print 'ntree limit:', ntree_limit
            print 'best_mcc:', best_mcc
            print 'best_proba:', best_proba
            print 'matthews_corroef', mcc_official

            run_info['ntree_limit_train'] = ntree_limit
            run_info['best_mcc_train'] = best_mcc
            run_info['best_proba_train'] = best_proba
            run_info['mcc_official_train'] = mcc_official

            ub.log('Predictions on xgval...', 'highlight')
            predictions = model.predict(xgval, ntree_limit=ntree_limit)

            best_proba, best_mcc, y_pred = ub.eval_mcc(y_val, predictions, True)
            mcc_official = matthews_corrcoef(y_val, y_pred)
            print 'ntree limit:', ntree_limit
            print 'best_mcc:', best_mcc
            print 'best_proba:', best_proba
            print 'matthews_corroef', mcc_official

            run_info['ntree_limit_val'] = ntree_limit
            run_info['best_mcc_val'] = best_mcc
            run_info['best_proba_val'] = best_proba
            run_info['mcc_official_val'] = mcc_official

            if analyze_feature_importance:
                ub.log('Analyzing feature importance...')
                feature_imp_fname_template = run_info['feature_imp_fname_template']
                top_features_fname = run_info['top_features_fname']
                feature_imp_fname = feature_imp_fname_template.format(postfix_train)
                imp = model.get_fscore()
                imp = sorted(imp.items(), key=operator.itemgetter(1))
                imp_df = pd.DataFrame(imp, columns=['feature', 'fscore'])
                imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()

                ub.log('Output result csv to {}...'.format(feature_imp_fname + '.csv'))
                imp_df.to_csv(feature_imp_fname + '.csv')

                plt.figure()
                imp_df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
                plt.title('XGBoost Feature Importance @ {}'.format(postfix_train))
                plt.xlabel('relative importance')
                plt.gcf().savefig(feature_imp_fname + '.png', bbox_inches='tight')

                feature_lists = glob.glob(feature_imp_fname_template.replace('{}', '*.csv'))
                ub.log('Aggregating previous analysis results...')
                print feature_lists
                features_df = None
                if feature_lists:
                    for f_l in feature_lists:
                        tmp_df = pd.read_csv(f_l, index_col=0)
                        if features_df is None:
                            features_df = tmp_df
                        else:
                            features_df = pd.concat([features_df, tmp_df], ignore_index=True)

                f_df = features_df.groupby(['feature']).mean().reset_index()
                f_df['overall'] = True
                imp_df['overall'] = False
                merged_df = pd.concat([imp_df, f_df]).sort_values(by=['overall', 'fscore'], ascending=False)
                sns_plot = sns.factorplot(y='feature', x='fscore', data=merged_df, hue='overall', kind='bar',
                                          hue_order=[True, False], size=20, aspect=0.5)
                sns_plot.savefig(feature_imp_fname + '_overall.png', bbox_inches='tight')

                ub.log('Output overall result csv to {}...'.format(top_features_fname))
                with open(top_features_fname, 'w') as tf:
                    tf.write('\n'.join(list(set(merged_df.feature.values))))

                merged_df.to_csv(top_features_fname.replace('.txt', '_df.csv'), index=False)


                # json has trouble serializing np.float32
                # with open(run_info_fname, 'w') as fp:
                #    json.dump(run_info, fp)

    if make_submission:

        if not train_model and not compile_data:
            assert (run_info_fname is not None)
            ub.log('(make_submission) Loading run info from {} ...'.format(run_info_fname))
            with open(run_info_fname, 'r') as fp:
                run_info = eval(fp.read())
            print json.dumps(run_info, indent=2)

        if ub.home_dir not in run_info['model_fname']:
            for i in ub.possible_home_dirs:
                if i in run_info['model_fname']:
                    logged_home_dir = i

        for k in ['fname_xgtest', 'model_fname']:
            if ub.home_dir not in run_info[k]:
                for i in ub.possible_home_dirs:
                    if i in run_info[k]:
                        run_info[k] = run_info[k].replace(i, ub.home_dir)

        if not train_model:
            model = xgb.Booster()
            ub.log('Loading model {} ...'.format(run_info['model_fname']))
            model.load_model(run_info['model_fname'])

        if not compile_data:
            ub.log('Loading xgtest data {} ...'.format(run_info['fname_xgtest']))
            xgtest = xgb.DMatrix(run_info['fname_xgtest'])

        ub.log('XGB making predictions...')

        postfix_train = run_info['postfix_train']

        ypred = model.predict(xgtest, ntree_limit=run_info['ntree_limit_train'])
        nrows = len(ypred)

        sample = pd.read_csv(os.path.join(ub.data_dir, 'sample_submission.csv'), nrows=nrows)
        sample['Response'] = ypred
        fname_output = os.path.join(ub.output_dir, "sub_xgboost_{}_prob.csv".format(postfix_train))
        if not compile_data:
            fname_output = fname_output.replace('.csv', '_{}.csv'.format(datetime_str))
        ub.log('Writing output file (raw proba) {} ...'.format(fname_output))
        sample.to_csv(fname_output, index=False)

        best_proba = (run_info['best_proba_train'] + run_info['best_proba_val']) / 2.0
        ub.log('Using threshold: best_proba == {}'.format(best_proba))
        sample['Response'] = (ypred > best_proba).astype(int)
        fname_output = os.path.join(ub.output_dir, "sub_xgboost_{}.csv".format(postfix_train))
        if not compile_data:
            fname_output = fname_output.replace('.csv', '_{}.csv'.format(datetime_str))
        ub.log('Writing output file {} ...'.format(fname_output))
        sample.to_csv(fname_output, index=False)

    if compile_data or train_model:
        if compile_data:
            if run_info_fname is not None:
                ub.log('Ignore input run_info_fname {}'.format(run_info_fname))
            run_info_fname = os.path.join(ub.output_dir, 'run_info_{}.txt'.format(postfix_train))
        # else run_info_fname is an input parameter
        ub.log('Saving run_info into {}'.format(run_info_fname))
        print pd.Series(run_info)
        with open(run_info_fname, 'w') as fp:
            fp.write(str(run_info))

    return run_info_fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-D', action='store_true',
                        default=False, dest='compile_data',
                        help='>> Set True to compile_data')

    parser.add_argument('-oco', action='store_true',
                        default=False, dest='original_cols_only',
                        help='>> Set True to original_cols_only')

    parser.add_argument('-disable_id_diff', action='store_true',
                        default=False, dest='disable_id_diff_cols',
                        help='>> Set True to disable_id_diff_cols')

    parser.add_argument('-ntrain', action='store', type=int,
                        default=1, dest='N_files_train',
                        help='>> Specifies number of partial train files to load')
    parser.add_argument('-ntest', action='store', type=int,
                        default=1, dest='N_files_test',
                        help='>> Specifies number of partial test files to load')
    parser.add_argument('-nstart', action='store', type=int,
                        default=0, dest='N_start',
                        help='>> Specifies start id of partial files')

    parser.add_argument('-ff', action='store',
                        default=None, dest='feature_list_file',
                        help='>> Specifies feature_list_file to use if any')

    parser.add_argument('-skip_date', action='store_true',
                        default=False, dest='skip_date_csv',
                        help='>> Set True to skip_date_csv')

    parser.add_argument('-skip_num', action='store_true',
                        default=False, dest='skip_num_csv',
                        help='>> Set True to skip_num_csv')

    parser.add_argument('-skip_cat', action='store_true',
                        default=False, dest='skip_cat_csv',
                        help='>> Set True to skip_cat_csv')

    parser.add_argument('-T', action='store_true',
                        default=False, dest='train_model',
                        help='>> Set True to train_model')

    parser.add_argument('-imp', action='store_true',
                        default=False, dest='imp',
                        help='>> Specifies if analyze feature analysis')
    parser.add_argument('-cv', action='store_true',
                        default=False, dest='cv',
                        help='>> Specifies if run cv')
    parser.add_argument('-nrounds', action='store', type=int,
                        default=1000, dest='N_rounds',
                        help='>> Specifies number of xgb rounds')
    parser.add_argument('-es', action='store', type=int,
                        default=10, dest='early_stop',
                        help='>> Specifies early_stop_rounds')

    default_xgb_params = get_params()

    parser.add_argument('-xgb_eta', action='store', type=float,
                        default=default_xgb_params['eta'], dest='xgb_eta',
                        help='>> Specifies eta')
    parser.add_argument('-xgb_md', action='store', type=int,
                        default=default_xgb_params['max_depth'], dest='xgb_md',
                        help='>> Specifies max_depth')
    parser.add_argument('-xgb_mcw', action='store', type=int,
                        default=default_xgb_params['min_child_weight'], dest='xgb_mcw',
                        help='>> Specifies min_child_weight')
    parser.add_argument('-xgb_ss', action='store', type=float,
                        default=default_xgb_params['subsample'], dest='xgb_ss',
                        help='>> Specifies subsample')
    parser.add_argument('-xgb_cs', action='store', type=float,
                        default=default_xgb_params['colsample_bytree'], dest='xgb_cs',
                        help='>> Specifies colsample_bytree')

    parser.add_argument('-S', action='store_true',
                        default=False, dest='make_submission',
                        help='>> Set True to make_submission')

    parser.add_argument('-run_info', action='store', type=str,
                        default=None, dest='run_info_fname',
                        help='>> Specifies run_info_fname')

    par = parser.parse_args()
    if par.run_info_fname is not None:
        par.run_info_fname = os.path.join(ub.output_dir, os.path.basename(par.run_info_fname))

    new_xgb_params = get_params()
    new_xgb_params['eta'] = par.xgb_eta
    new_xgb_params['max_depth'] = par.xgb_md
    new_xgb_params['min_child_weight'] = par.xgb_mcw
    new_xgb_params['subsample'] = par.xgb_ss
    new_xgb_params['colsample_bytree'] = par.xgb_cs

    ub.log('Input parameters:', 'info')
    # print pd.Series(par.__dict__)

    par_dict = dict(
        compile_data=par.compile_data,
        N_files_train=par.N_files_train,
        N_files_test=par.N_files_test,
        N_start=par.N_start,
        feature_list_file=par.feature_list_file,
        original_cols_only=par.original_cols_only,
        disable_id_diff_cols=par.disable_id_diff_cols,
        skip_date_csv=par.skip_date_csv,
        skip_num_csv=par.skip_num_csv,
        skip_cat_csv=par.skip_cat_csv,

        train_model=par.train_model,
        N_rounds=par.N_rounds,
        cv=par.cv,
        analyze_feature_importance=par.imp,
        early_stop_rounds=par.early_stop,

        make_submission=par.make_submission,
        run_info_fname=par.run_info_fname,
        xgb_params=new_xgb_params
    )
    print pd.Series(par_dict)

    run_info_fname = main(**par_dict)
