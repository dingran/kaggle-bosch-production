import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
import os
import datetime
import util_bosch as ub
import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_params(base_core=None):
    if base_core is None:
        base_core = 0.0058
    # https://github.com/dmlc/xgboost/blob/master/doc/param_tuning.md
    params = {
        # 1- General Parameters
        # 'booster': "gbtree",  # booster [default=gbtree]
        'silent': True,  # silent [default=0]
        # 'nthread' : 8 , #nthread [default to maximum number of threads available if not set]

        # 2A-Parameters for Tree Booster
        'eta': 0.023,  # 0.023, # eta [default=0.3] range: [0,1]
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
        'base_score': base_core,  # base_score [ default=0.5 ]
        # 'eval_metric': 'logloss',  # eval_metric [ default according to objective ]
        'seed': int(np.random.rand() * 100000)  # seed [ default=0 ]

    }
    return params


use_buffer = False

upsampling = 2  # None to turn it off
downsampling = None #None  # specify portion of response==0 samples to keep

make_submission = False
do_cv = True
do_validation = True
report_feature_imp = False

holdout_set = True
out_of_fold_averaging = False
if out_of_fold_averaging:
    holdout_set=True

#NROWS = 500000
NROWS = None
holdout_size = 0.2
testsize = 0.2

N_rounds = 200
N_early = 10

tag = 'v1'
sub_str = ''
# sub_str = '_sub'
# sub_str = '_tiny'

xgb_params = get_params()


def OutOfFoldAverage(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['Response'].mean().reset_index()
    grpCount = data1.groupby(columnName)['Response'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.Response
    if (useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 1]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['Response'].values
    x = pd.merge(data2[[columnName, 'Response']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['Response']
    if (useLOO):
        x = ((x * x.shape[0]) - outcomes) / (x.shape[0] - 1)
        #  x = x + np.random.normal(0, .01, x.shape[0])
    return x.fillna(x.mean())


# %%
if use_buffer:
    xgtrain_fname = '../data_processed/xgtrain{}_{}.buffer'.format(sub_str, tag)

    with open('../data_processed/xgb_features{}_{}.txt'.format(sub_str, tag), 'r') as ff:
        feature_names = [x.strip() for x in ff.readlines()]
    ub.log('Loading xgtrain DMatrix...{}'.format(xgtrain_fname))
    xgtrain = xgb.DMatrix(xgtrain_fname, feature_names=feature_names)

else:
    df_train_fname = '../data_processed/df_train_overall_{}.csv'.format(tag)
    # df_train_fname = '../data/train_numeric.csv'
    df_train2 = pd.read_csv(df_train_fname, nrows=1)
    feature_list_fname = 'importance_ordered_list_1107.txt'
    #feature_list_fname = 'feature_select_1102.txt'
    with open(feature_list_fname, 'r') as ff:
        selected_features = [fe.strip() for fe in ff.readlines()]

    OOFA_encode_features = ['start_date', 'start_date_id_diff', 'start_date_id_diff_reverse',
                            'L1_S24_F1559', 'L3_S32_F3851', 'L1_S24_F1827', 'L1_S24_F1582',
                            'L3_S32_F3854', 'L1_S24_F1510', 'L1_S24_F1525', 'L3_S30_start_date', 'L3_S30_D3496',
                            'L3_S30_D3506', 'L3_S30_D3501', 'L3_S30_D3516', 'L3_S30_D3511', 'L3_S33_D3856']

    selected_features = ['Id', 'Response', 'start_date_diff'] + selected_features

    if 1:
        feature_selected = selected_features[:50]

#        orig_f= [x for x in feature_selected if x.startswith('L') and (x.endswith('0') or x.endswith('1') or x.endswith('2') or x.endswith('3') or x.endswith('4') or x.endswith('5') or x.endswith('6') or x.endswith('7') or x.endswith('8') or x.endswith('9'))]
#        for o in orig_f:
#            feature_selected.remove(o)
#
#        feature_selected = feature_selected + ['L1_S24_F1559', 'L3_S32_F3851', 'L1_S24_F1827', 'L1_S24_F1582',
#         'L3_S32_F3854', 'L1_S24_F1510',
#         'L1_S24_F1525', 'L3_S30_D3496', 'L3_S30_D3506',
#         'L3_S30_D3501', 'L3_S30_D3516',
#         'L3_S30_D3511','L3_S33_D3856',
#         'L0_S0_F20','L3_S30_F3759',
#         'L3_S30_F3749','L0_S0_F0',
#         'L0_S0_F16',
#         'L1_S24_F1846', 'L3_S32_F3850',
#         'L1_S24_F1695', 'L1_S24_F1632',
#         'L3_S33_F3855', 'L1_S24_F1604',
#         'L3_S29_F3407', 'L3_S33_F3865',
#         'L3_S38_F3952', 'L1_S24_F1723']

        xx = [x for x in feature_selected if x not in df_train2.columns]
        for i in xx:
            feature_selected.remove(i)

        xx = [x for x in OOFA_encode_features if x not in df_train2.columns]
        for i in xx:
            OOFA_encode_features.remove(i)

        if out_of_fold_averaging:
            feature_selected = list(set(feature_selected + OOFA_encode_features))
        
        feature_selected = list(set(feature_selected))
    else:
        feature_selected = list(df_train2.columns)

    df_train = pd.read_csv(df_train_fname, usecols=feature_selected, nrows=NROWS)
    #df_train = df_train.merge(pd.read_csv('train_ex.csv', usecols=['Id', 'test_1', 'test_2', 'test_3', 'test_4']), on='Id')
    #df_train = df_train[['test_1', 'test_2', 'test_3', 'test_4'] + feature_selected]
    df_train = df_train[feature_selected]

    if holdout_set:

        print 'holdout set size = {}'.format(holdout_size)
        shuffle_col = df_train[['Id']].copy()
        shuffle_col['Id'] = np.random.rand(len(shuffle_col))
        train_idx = shuffle_col[shuffle_col['Id'] > holdout_size].index
        holdout_idx = shuffle_col[shuffle_col['Id'] <= holdout_size].index
        
        df_holdout = df_train.loc[holdout_idx]
        df_train = df_train.loc[train_idx]
        
        #df_holdout = df_train.loc[::2]
        #df_train = df_train.loc[1::2]
        print 'train: ', df_train.shape
        print 'holdout: ', df_holdout.shape

        if out_of_fold_averaging:
            for col in OOFA_encode_features:
                print(col)
                df_train.loc[:, col + '_OOF'] = OutOfFoldAverage(df_holdout, df_train, col, False).values
            oof_cols = [c for c in df_train.columns if 'OOF' in c]
            df_train[['Id']+OOFA_encode_features+oof_cols].head(1000).to_csv('debug_oof.csv')
        else:
            y_holdout = df_holdout.Response.values
            df_holdout.drop(['Response'], axis=1, inplace=True)
            xgholdout = xgb.DMatrix(df_holdout, label=y_holdout, feature_names=df_holdout.columns)

    shuffle_col = df_train[['Id']].copy()
    shuffle_col['Id'] = np.random.rand(len(shuffle_col))

    train_idx = shuffle_col[shuffle_col['Id'] > testsize].index
    val_idx = shuffle_col[shuffle_col['Id'] <= testsize].index
    ub.log('Done shuffling...')
    print 'len of train_idx', len(train_idx)
    print 'len of val_idx', len(val_idx)

    df_val = df_train.loc[val_idx]
    df_train = df_train.loc[train_idx]

    print df_train.shape
    print 'Train set positive sample ratio: {}/{}'.format(df_train.Response.sum(), len(df_train))
    print df_val.shape
    print 'Validation set positive sample ratio: {}/{}'.format(df_val.Response.sum(), len(df_val))

    if upsampling is not None:  # up sampling
        print 'Upsampling == {}'.format(upsampling)
        df_train_resp1 = df_train[df_train['Response'] == 1].copy()
        for i in range(upsampling):
            df_train = pd.concat([df_train, df_train_resp1], ignore_index=True)
        print '(new) Train set positive sample ratio: {}/{}'.format(df_train.Response.sum(), len(df_train))

    if downsampling is not None:
        print 'Downsampling == {}'.format(downsampling)

        df_train_resp1 = df_train[df_train['Response'] == 1].copy()
        df_train = df_train[df_train['Response'] == 0]
        shuffle_col = df_train[['Id']].copy()
        shuffle_col['Id'] = np.random.rand(len(shuffle_col))
        keep_idx = shuffle_col[shuffle_col['Id'] < downsampling].index
        df_train = df_train.loc[keep_idx]
        df_train = pd.concat([df_train, df_train_resp1], ignore_index=True)
        print '(new) Train set positive sample ratio: {}/{}'.format(df_train.Response.sum(), len(df_train))

    y_train = df_train.Response.values
    df_train.drop(['Response'], axis=1, inplace=True)
    xgtrain = xgb.DMatrix(df_train, label=y_train, feature_names=df_train.columns)

    y_val = df_val.Response.values
    df_val.drop(['Response'], axis=1, inplace=True)
    xgval = xgb.DMatrix(df_val, label=y_val, feature_names=df_val.columns)

print xgtrain.num_row(), xgtrain.num_col()
print xgval.num_row(), xgval.num_col()
# xgtrain = xgtrain.slice(range(100))

# %%

eval_hist = None
if do_cv:
    ub.log('Running cross validation...')
    #xgb_params['eval_metric'] = ['auc'] 
    eval_hist = xgb.cv(xgb_params, xgtrain, num_boost_round=N_rounds,
                       early_stopping_rounds=N_early,
                       feval=ub.mcc_eval, maximize=True,
                       verbose_eval=1, show_stdv=True, nfold=3, seed=0, stratified=True)
    #xgb_params.pop('eval_metric')
    # print eval_hist

# %%

if eval_hist is not None:
    num_boost_round = len(eval_hist) + N_early
    num_early = len(eval_hist)
else:
    num_boost_round = N_rounds
    num_early = N_early

if do_validation:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ub.log('Running training...')
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(xgb_params, xgtrain, num_boost_round=num_boost_round,
                      early_stopping_rounds=num_early,
                      feval=ub.mcc_eval, maximize=True,
                      evals=watchlist, verbose_eval=True)

    model_fname = os.path.join(ub.output_dir, 'xbg_{}.model'.format(datetime_str))
    ub.log('Saving model: {}...'.format(model_fname))
    model.save_model(model_fname)
    model.dump_model(model_fname + '.raw.txt')

    ntree_limit = model.best_iteration + 1

    ub.log('Predictions on xgtrain... ntree_limit = {}'.format(ntree_limit), 'highlight')
    predictions = model.predict(xgtrain, ntree_limit=ntree_limit)
    best_proba_train, best_mcc, y_pred = ub.eval_mcc(y_train, predictions, True)
    print 'Overall positive ratio: {}'.format(y_pred.sum() * 1.0 / len(y_pred))
    mcc_official = matthews_corrcoef(y_train, y_pred)
    print 'best_mcc:', best_mcc
    print 'best_proba:', best_proba_train
    print 'matthews_corroef', mcc_official

    ub.log('Predictions on xgval... ntree_limit = {}'.format(ntree_limit), 'highlight')
    predictions = model.predict(xgval, ntree_limit=ntree_limit)
    best_proba_val, best_mcc, y_pred = ub.eval_mcc(y_val, predictions, True)
    print 'Overall positive ratio: {}'.format(y_pred.sum() * 1.0 / len(y_pred))
    mcc_official = matthews_corrcoef(y_val, y_pred)
    print 'best_mcc:', best_mcc
    print 'best_proba:', best_proba_val
    print 'matthews_corroef', mcc_official

    if holdout_set and not out_of_fold_averaging:
        ub.log('Predictions on xgholdout... ntree_limit = {}'.format(ntree_limit), 'highlight')
        predictions = model.predict(xgholdout, ntree_limit=ntree_limit)
        #        best_proba, best_mcc, y_pred = ub.eval_mcc(y_val, predictions, True)
        #        print 'Overall positive ratio: {}'.format(y_pred.Response.sum()*1.0/len(y_pred))
        #        mcc_official = matthews_corrcoef(y_val, y_pred)
        #        print 'best_mcc:', best_mcc
        #        print 'best_proba:', best_proba
        #        print 'matthews_corroef', mcc_official

        print 'Result using best_proba ({}) from train set:'.format(best_proba_train)
        y_pred = predictions > best_proba_train
        print 'Overall positive ratio: {}'.format(y_pred.sum() * 1.0 / len(y_pred))
        print 'MCC: ',
        print matthews_corrcoef(y_holdout, y_pred)

        print 'Result using best_proba ({}) from val set:'.format(best_proba_val)
        y_pred = predictions > best_proba_val
        print 'Overall positive ratio: {}'.format(y_pred.sum() * 1.0 / len(y_pred))
        print 'MCC: ',
        print matthews_corrcoef(y_holdout, y_pred)

        print 'Using optimized best_proba for holdout set'
        best_proba_holdout, best_mcc, y_pred = ub.eval_mcc(y_holdout, predictions, True)
        print 'Overall positive ratio: {}'.format(y_pred.sum() * 1.0 / len(y_pred))
        mcc_official = matthews_corrcoef(y_holdout, y_pred)
        print 'best_mcc:', best_mcc
        print 'best_proba:', best_proba_holdout
        print 'matthews_corroef', mcc_official

    ub.log('Analyzing feature importance...')
    feature_imp_fname_template = 'feature_importance_xgb_{}'
    feature_imp_fname = feature_imp_fname_template.format(datetime_str)
    imp = model.get_fscore()
    imp = sorted(imp.items(), key=operator.itemgetter(1))
    imp_df = pd.DataFrame(imp, columns=['feature', 'fscore'])
    imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()

    plt.figure()
    imp_df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 80. / 420 * len(imp_df)))
    plt.title('XGBoost Feature Importance @ {}'.format(datetime_str))
    plt.xlabel('relative importance')

    if report_feature_imp:
        ub.log('Output result csv to {}...'.format(feature_imp_fname + '.csv'))
        imp_df.to_csv(feature_imp_fname + '.csv')
        plt.gcf().savefig(feature_imp_fname + '.png', bbox_inches='tight')

# %%
if not make_submission:
    sys.exit(0)

# %%

if use_buffer:
    xgtest_fname = '../data_processed/xgtest{}_{}.buffer'.format(sub_str, tag)
    ub.log('Loading xgtest DMatrix...{}'.format(xgtest_fname))
    xgtest = xgb.DMatrix(xgtest_fname, feature_names=feature_names)
else:
    df_test_fname = '../data_processed/df_test_overall_{}.csv'.format(tag)
    if 'Response' in feature_selected:
        feature_selected.remove('Response')
    df_test = pd.read_csv(df_test_fname, usecols=feature_selected)
    #df_test = df_test.merge(pd.read_csv('test_ex.csv', usecols=['Id', 'test_1', 'test_2', 'test_3', 'test_4']), on='Id')
    #df_test = df_test[['test_1', 'test_2', 'test_3', 'test_4'] + feature_selected]
    df_test = df_test[feature_selected]
    if out_of_fold_averaging:
        for col in OOFA_encode_features:
            print(col)
            df_test.loc[:, col + '_OOF'] = OutOfFoldAverage(df_holdout, df_test, col, False).values
    xgtest = xgb.DMatrix(df_test, feature_names=df_test.columns)

print xgtest.num_row()
print xgtest.num_col()

# %%
ub.log('Making prediction on xgtest...')
# model = xgb.Booster()
# ub.log('Loading model {} ...'.format(model_fname))
# model.load_model(model_fname)
ypred = model.predict(xgtest, ntree_limit=ntree_limit)

sample = pd.read_csv(os.path.join(ub.data_dir, 'sample_submission.csv'))
sample['Response'] = ypred
fname_output = os.path.join(ub.output_dir, "sub_xgboost_{}_prob.csv".format(datetime_str))

ub.log('Writing output file (raw proba) {} ...'.format(fname_output))
sample.to_csv(fname_output, index=False)

best_proba = best_proba_train
if upsampling or downsampling:
    best_proba = best_proba_val

ub.log('Using threshold: best_proba == {}'.format(best_proba))
sample['Response'] = (ypred > best_proba).astype(int)
print 'Overall positive ratio: {}'.format(sample.Response.sum() * 1.0 / len(ypred))
fname_output = os.path.join(ub.output_dir, "sub_xgboost_{}_with_best_proba.csv".format(datetime_str))
ub.log('Writing output file {} ...'.format(fname_output))
sample.to_csv(fname_output, index=False)

sys.exit(0)
# %%

tol = 0.0001
target_positive_rate = 0.0058
mp0 = best_proba
step = 0.05
positive_rate = sum(ypred > mp0) * 1.0 / len(ypred)
current_sign = np.sign(positive_rate - target_positive_rate)
i_iter = 0
while (np.abs(positive_rate - target_positive_rate) > tol):
    mp0 += step * current_sign
    previous_sign = current_sign
    positive_rate = sum(ypred > mp0) * 1.0 / len(ypred)
    current_sign = np.sign(positive_rate - target_positive_rate)
    if previous_sign != current_sign:
        step = step / 2
    print '[{}], proba {}, stepsize {}, positive_rate {}'.format(i_iter, mp0, step, positive_rate)
    i_iter += 1

manual_proba = mp0
# %%
pr = []
mr = np.arange(0.05, 0.4, 0.01)
for m in mr:
    pr.append(sum(ypred > m) * 1.0 / len(ypred))

plt.plot(mr, pr)

# %%
manual_proba = 0.2
positive_rate = sum(ypred > manual_proba) * 1.0 / len(ypred)
ub.log('Using threshold: manual_proba == {}, positive_rate == {}'.format(manual_proba, positive_rate))
sample['Response'] = (ypred > manual_proba).astype(int)
fname_output = os.path.join(ub.output_dir,
                            "sub_xgboost_{}_with_proba={:.4f}_rate={:.4f}.csv".format(datetime_str, manual_proba,
                                                                                      positive_rate))
ub.log('Writing output file {} ...'.format(fname_output))
sample.to_csv(fname_output, index=False)
