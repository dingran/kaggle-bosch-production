import util_bosch as ub
import pandas as pd
import os
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker, Timer


def split_csv(basefname='train_numeric.csv'):
    basefname_part_template = ub.data_fname_to_partial_data_fname_template(basefname)
    input_fname = os.path.join(ub.data_dir, basefname)
    output_fname_template = os.path.join(ub.data_dir, basefname_part_template)

    ub.log('reading {}'.format(input_fname))
    sw = ub.StopWatch('read ' + basefname)
    reader = pd.read_csv(input_fname, chunksize=50000, low_memory=False)
    for i, chuck in enumerate(reader):
        print i
        chuck.to_csv(output_fname_template.format(i), index=False)
    ub.log(sw.stop())


def split_csv_old(basefname='train_numeric.csv'):
    basefname_part_template = ub.data_fname_to_partial_data_fname_template(basefname)
    input_fname = os.path.join(ub.data_dir, basefname)
    output_fname_template = os.path.join(ub.data_dir, basefname_part_template)

    ub.log('reading {}'.format(input_fname))
    nrows = None
    sw = ub.StopWatch('read ' + basefname)
    df_train = pd.read_csv(input_fname, nrows=nrows, low_memory=False)
    print 'df shape: {}'.format(df_train.shape)
    ub.log(sw.stop())

    # split original csv
    ub.log('splitting {}'.format(basefname))

    N_split = ub.N_split

    ub.log('outputting {}'.format(output_fname_template))
    n_rows = int(df_train.shape[0] / N_split)

    widgets = ['splitting', ': ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=N_split).start()
    for i in range(N_split):
        output_fname = output_fname_template.format(i)
        if 0:
            if i < N_split - 1:
                print 'writng rows:', i * n_rows, (i + 1) * n_rows - 1
            else:
                print 'writng rows:', i * n_rows, df_train.shape[0]

        df_train[i * n_rows:(i + 1) * n_rows].to_csv(output_fname, index=False)
        pbar.update(i)
    pbar.finish()


if __name__ == '__main__':
    f_list = [
        'train_numeric.csv',
        'train_categorical.csv',
        'train_date.csv',
        'test_numeric.csv',
        'test_categorical.csv',
        'test_date.csv',
    ]

    for f in f_list:
        ub.log('*' * 50 + f, 'highlight')
        split_csv(f)
