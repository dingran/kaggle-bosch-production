import pandas as pd
import os
import platform
import logging
import matplotlib.pyplot as plt


import numpy as np 
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(format="%(asctime)s %(levelname)s: (in %(pathname)s) %(message)s", 
                    level=logging.DEBUG)

logger = logging.getLogger()



# import data sets

home_dir = '/Users/ran/'
if platform.system() == 'Linux':
    home_dir = '/home/dingran/'
    if os.path.exists('/storage/'):
        home_dir = '/storage/dingran/'

data_dir = os.path.join(home_dir, 'Dropbox/Kaggle/bosch-production/data')

nrows = None
nrows = 1000

nrows_test = None
nrows_test = 1000

df_train_num = pd.read_csv(os.path.join(data_dir, 'train_numeric.csv'), nrows=nrows)
df_test_num = pd.read_csv(os.path.join(data_dir, 'test_numeric.csv'), nrows=nrows_test)

df_train_cat = pd.read_csv(os.path.join(data_dir, 'train_categorical.csv'), nrows=nrows)
df_test_cat = pd.read_csv(os.path.join(data_dir, 'test_categorical.csv'), nrows=nrows_test)

df_train_date = pd.read_csv(os.path.join(data_dir, 'train_date.csv'), nrows=nrows)
df_test_date = pd.read_csv(os.path.join(data_dir, 'test_date.csv'), nrows=nrows_test)


#%%

#df_train = pd.merge(df_train_num, df_train_cat, how='inner', left_on='Id', right_on='Id')
#df_test = pd.merge(df_test_num, df_test_cat, how='inner', left_on='Id', right_on='Id')

df_train = df_train_num
df_test = df_test_num

df_train['is_train'] = 1
df_test['is_train'] = 0

#df_train = df_train[0:df_train.shape[0]]
#df_test = df_test[0:10]

print df_train.shape
print df_test.shape

df = pd.concat([df_train, df_test])

#%% find high cardinal fields

def col_check(df):
    del df['Id']
    df_c = pd.Series()
    for c in df.columns:
        if len(df[c].unique()) > 1:
            #print c, len(df[c].unique()), df[c].dtype,
            #print df[c].unique()
            pass
        df_c[c]=len(df[c].unique())
    #plt.figure()
    #df_c.hist()
    
    print df_c.shape[0], 'columns'
    print sum(df_c==1), 'all nan columns'
    print sum(df_c>1), 'non all nan columns'
    print sum(df_c>10), 'high cardinal columns'
    print df_c.describe()
    
#%%
col_check(df_train_num)   
col_check(df_train_cat)   
col_check(df_train_date)    


#%%
cols_full_flag = df.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))
print non_full_cols

#%%
if 1:
    df.fillna(-1, inplace=True)
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
            
#%%
cols_full_flag = df.isnull().any()
non_full_cols = list(cols_full_flag[cols_full_flag].index)
print 'Non-full columns: {}'.format(len(non_full_cols))


#%%
le = LabelEncoder()
obj_cols = df.select_dtypes(include=['object']).columns
print 'Obj columns: ', list(obj_cols)
for col in obj_cols:
    df[col] = le.fit_transform(df[col])

#%%
import datetime

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
columns_to_drop = ['Id', 'Response', 'is_train']
xgb_num_rounds = 2000
feature_names = list(df.drop(columns_to_drop, axis=1).columns)

df_train = df[df.is_train == 1]
df_test = df[df.is_train == 0]

testsize = 0.3

x_train, x_val, y_train, y_val = train_test_split(df_train.drop(columns_to_drop, axis=1).values, df_train['Response'].values, test_size=testsize)

#%%
from sklearn.metrics import matthews_corrcoef

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    thresholds = np.linspace(0.01, 0.99, 50)
    mcc = np.array([matthews_corrcoef(labels, preds>thr) for thr in thresholds])
    best_score = mcc.max()
    return 'error', -best_score
    
#%%
    
 
def get_params():
    seed = 1019
    # https://github.com/dmlc/xgboost/blob/master/doc/param_tuning.md
    param =     {
        #1- General Parameters       
        'booster' : "gbtree", #booster [default=gbtree]
        'silent': 0 , #silent [default=0]
        #'nthread' : 8 , #nthread [default to maximum number of threads available if not set]
    
        #2A-Parameters for Tree Booster   
        'eta'  :0.02, #0.023, # eta [default=0.3] range: [0,1]
        #'gamma':0 ,#gamma [default=0] range: [0,inf]
        'max_depth'           :6, #max_depth [default=6] range: [1,inf]
        #'min_child_weight':1,  #default=1]range: [0,inf]
        #'max_delta_step':0, #max_delta_step [default=0] range: [0,inf]
        'subsample'           :0.7, #0.83, #subsample [default=1]range: (0,1]
        'colsample_bytree'    :0.77, #0.77, #colsample_bytree [default=1]range: (0,1]
        #'lambda': 1,  #lambda [default=1]
        #'alpha':0.0001, #alpha [default=0]
        
        
        #2B- Parameters for Linear Booster
        #'lambda': 0,  #lambda [default=0]
        #'alpha':0, #alpha [default=0]
        #'lambda_bias':0, #default 0
        
        #3- earning Task Parameters
        'objective': 'binary:logistic',  #objective [ default=reg:linear ]
        #'base_score'=0.5,        #base_score [ default=0.5 ]
        #'eval_metric' : 'logloss', #eval_metric [ default according to objective ]
        'seed':seed #seed [ default=0 ]
      
        }
    return param
    
    
xgtrain = xgb.DMatrix(x_train, y_train, feature_names=feature_names)
xgval = xgb.DMatrix(x_val, y_val, feature_names=feature_names)

plst = get_params()
print(plst)


cv = 0
if cv:
    eval_hist = xgb.cv(plst, xgtrain, xgb_num_rounds, nfold=3, show_progress=True, early_stopping_rounds=50)
    eval_hist.plot()
    plt.figure()
    plt.plot(eval_hist['test-logloss-mean'])
else:
    evallist = [(xgtrain, 'train'), (xgval, 'eval')]
    model = xgb.train(plst, xgtrain, xgb_num_rounds, evals=evallist, early_stopping_rounds=50) #, feval=evalerror)
    model.save_model('xbg-' + datetime_str + '.model')


    if 1:
        print 'XGB making predictions'
        x_test = df_test.drop(columns_to_drop, axis=1).values
        dtest = xgb.DMatrix(x_test, feature_names=feature_names)
        # ypred = bst.predict(dtest)
        ypred = model.predict(dtest, ntree_limit=model.best_iteration)
    
        sample = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'), nrows=nrows_test)
        sample['Response'] = ypred
        sample.to_csv("sub_xgboost" + datetime_str + ".csv", index=False)
