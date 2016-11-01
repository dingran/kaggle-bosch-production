import os
import sys
import platform
import termcolor
import datetime
import time
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

N_split = 24


def data_fname_to_partial_data_fname_template(fname):
    return fname.replace('.', '_part{}.')


class StopWatch:
    def __init__(self, name=''):
        self.start_time = time.time()
        self.name = name

    def stop(self):
        t_elapsed = str(int((time.time() - self.start_time) * 100) / 100.0)
        return '({}): time elapsed: {}s.'.format(self.name, t_elapsed)


def get_datetime_str():
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime_str


def log(message='\n', kind='general', color_str=None):
    if color_str is None:
        if kind == 'error' or kind.startswith('e'):
            color_str = 'red'
        elif kind == 'info' or kind.startswith('i'):  # highlighted sys info message
            color_str = 'yellow'
        elif kind == 'overwrite' or kind.startswith('o'):  # overwrite related messages
            color_str = 'magenta'
        elif kind == 'write' or kind.startswith('w'):  # write files
            color_str = 'cyan'
        elif kind == 'highlight' or kind.startswith('h'):  # progress message highlighting
            color_str = 'green'
        else:
            color_str = 'white'  # normal progress/log messages

    print termcolor.colored(datetime.datetime.now(), color_str),
    print message
    sys.stdout.flush()


possible_home_dirs = ['/Users/ran/', '/home/dingran/', '/storage/dingran/']
home_dir = '/Users/ran/'
if platform.system() == 'Linux':
    home_dir = '/home/dingran/'
    if os.path.exists('/storage/'):
        home_dir = '/storage/dingran/'

project_dir = os.path.join(home_dir, 'Dropbox/Kaggle/bosch-production/')
data_dir = os.path.join(project_dir, 'data')
processed_data_dir = os.path.join(project_dir, 'data_processed')
if not os.path.exists(processed_data_dir):
    log('{} does not exist'.format(processed_data_dir))
    processed_data_dir = os.path.join(home_dir, 'data_processed')
    log('redirect to {}'.format(processed_data_dir))
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

code_dir = os.path.join(project_dir, 'code')
output_dir = os.path.join(project_dir, 'output')

train_files = [
    'train_date.csv',
    'train_numeric.csv',
    'train_categorical.csv'
]

test_files = [
    'test_date.csv',
    'test_numeric.csv',
    'test_categorical.csv'
]

cols = {'numeric': ['Id',
                    'L1_S24_F1846', 'L3_S32_F3850',
                    'L1_S24_F1695', 'L1_S24_F1632',
                    'L3_S33_F3855', 'L1_S24_F1604',
                    'L3_S29_F3407', 'L3_S33_F3865',
                    'L3_S38_F3952', 'L1_S24_F1723',
                    'Response'],
        'categorical': ['Id',
                        'L1_S24_F1559', 'L3_S32_F3851',
                        'L1_S24_F1827', 'L1_S24_F1582',
                        'L3_S32_F3854', 'L1_S24_F1510',
                        'L1_S24_F1525'],
        'date': ['Id',
                 'L3_S30_D3496', 'L3_S30_D3506',
                 'L3_S30_D3501', 'L3_S30_D3516',
                 'L3_S30_D3511']}


def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc
