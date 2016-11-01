import filecmp
import util_bosch as ub
import glob
import os
import pandas as pd
import re
import seaborn as sns
import datetime

do_plot = False

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
feature_imp_fname_template = os.path.join(ub.output_dir, 'feature_importance_xgb_{}')

feature_lists = glob.glob(feature_imp_fname_template.replace('{}', '*.csv'))
ub.log('Aggregating previous analysis results...')
print feature_lists
features_df = None
find_info = re.compile(r'feature_importance_xgb_(.*)_(\d+)of24.csv')
if feature_lists:
    for f_l in feature_lists:
        if 'accumu' in f_l:
            print 'skip ' + f_l
            continue
        tmp_df = pd.read_csv(f_l, index_col=0)
        results = find_info.search(f_l)
        datetime_info = results.group(1)
        n_datasets = results.group(2)
        fname = os.path.basename(f_l)
        id_info = '{}_{}sets'.format(datetime_info, n_datasets)
        # tmp_df = tmp_df.rename(columns={'fscore': })
        tmp_df['id_info'] = id_info

        if features_df is None:
            features_df = tmp_df
        else:
            # features_df = pd.merge(features_df, tmp_df, how='outer', on='feature')
            features_df = pd.concat([features_df, tmp_df], ignore_index=True)

ub.log('Generating plot')
order_list = features_df.groupby(['feature']).mean().sort_values(['fscore'], ascending=False).index
ar = 50 * len(order_list) / 1000

if do_plot:
    g = sns.factorplot(x='feature', y='fscore', data=features_df, kind='box', orient='v', aspect=ar, order=order_list)
    g.set_xticklabels(rotation=90)
    output_fname = os.path.join(ub.code_dir, 'feature_analysis.png')
    g.fig.suptitle('{} features, created {}'.format(len(order_list), datetime_str))
    g.savefig(output_fname, bbox_inches='tight')

ub.log('Done')

order_list = list(order_list)
output_fname = os.path.join(ub.code_dir, 'importance_ordered_list.txt')
ub.log('Writing ordered list to {}'.format(output_fname), 'write')
print 'N_features', len(order_list)
with open(output_fname, 'w') as fp:
    fp.write('\n'.join(order_list))

actual_features = []
for feature in order_list:
    if 'id_diff' in feature:
        # print feature
        actual_features.append(
            feature.replace('_id_diff', '').replace('_reverse', '').replace('_magic', ''))
        # these are created after loading in data
    else:
        actual_features.append(feature)
# print actual_features
# print len(actual_features)
# print len(set(actual_features))
actual_features = list(set(actual_features))


output_fname_template = os.path.join(ub.code_dir, 'prelim_feature_list_{}.txt')
prelim_feature_list_files = glob.glob(output_fname_template.replace('{}', '*'))
if prelim_feature_list_files:
    latest = max(prelim_feature_list_files)
    ub.log('Found latest prelim_feature_list to be {}'.format(latest), 'highlight')

output_fname = output_fname_template.format(datetime_str)
ub.log('Writing actual feature list to {}'.format(output_fname), 'write')
print 'N_features', len(actual_features)
with open(output_fname, 'w') as fp:
    fp.write('\n'.join(actual_features))
# print '\n'.join(actual_features)
if prelim_feature_list_files:
    if filecmp.cmp(latest, output_fname):
        ub.log('New file {} is identical to latest file {}'.format(output_fname, latest), 'highlight')
        ub.log('Deleting new file {}'.format(output_fname), 'overwrite')
        os.remove(output_fname)

# %%


cols = [['L1_S24_F1559', 'L3_S32_F3851',
         'L1_S24_F1827', 'L1_S24_F1582',
         'L3_S32_F3854', 'L1_S24_F1510',
         'L1_S24_F1525'],
        ['L3_S30_D3496', 'L3_S30_D3506',
         'L3_S30_D3501', 'L3_S30_D3516',
         'L3_S30_D3511'],
        ['L1_S24_F1846', 'L3_S32_F3850',
         'L1_S24_F1695', 'L1_S24_F1632',
         'L3_S33_F3855', 'L1_S24_F1604',
         'L3_S29_F3407', 'L3_S33_F3865',
         'L3_S38_F3952', 'L1_S24_F1723']]

hypercube_list = []

for i in cols:
    hypercube_list = hypercube_list + i

hypercube_list = list(set(hypercube_list))

ub.log('Merging hypercube list with prelim feature list')
for h in hypercube_list:
    if h not in actual_features:
        print 'adding', h
        actual_features.append(h)

output_fname = os.path.join(ub.code_dir, 'hypercube_list.txt')

with open(output_fname, 'w') as fp:
    fp.write('\n'.join(hypercube_list))


with open(os.path.join(ub.code_dir, 'FS_new_cols_only_20161030.txt'), 'r') as fp:
    new_cols = [x.strip() for x in fp.readlines()]

ub.log('Reading new cols list')
print 'N_new_cols', len(new_cols)

ub.log('Converting to actual new cols list')
actual_new_cols = []
for nc in new_cols:
    if 'id_diff' in nc:
        # print feature
        actual_new_cols.append(
            nc.replace('_id_diff', '').replace('_reverse', '').replace('_magic', ''))
        # these are created after loading in data
    else:
        actual_new_cols.append(nc)
actual_new_cols= list(set(actual_new_cols))
print 'N_new_cols', len(actual_new_cols)

ub.log('Merging actual_new_cols list with prelim feature list')

for n in actual_new_cols:
    if n not in actual_features:
        print 'adding', n
        actual_features.append(n)

output_fname_template = os.path.join(ub.code_dir, 'final_feature_list_{}.txt')
final_feature_list_files = glob.glob(output_fname_template.replace('{}', '*'))
if final_feature_list_files:
    latest = max(final_feature_list_files)
    ub.log('Found latest final_feature_list to be {}'.format(latest), 'highlight')

output_fname = output_fname_template.format(datetime_str)
ub.log('Writing final actual feature list to {}'.format(output_fname), 'write')
print 'N_features', len(actual_features)
with open(output_fname, 'w') as fp:
    fp.write('\n'.join(actual_features))
# print '\n'.join(actual_features)
if final_feature_list_files:
    if filecmp.cmp(latest, output_fname):
        ub.log('New file {} is identical to latest file {}'.format(output_fname, latest), 'highlight')
        ub.log('Deleting new file {}'.format(output_fname), 'overwrite')
        os.remove(output_fname)

# %%
