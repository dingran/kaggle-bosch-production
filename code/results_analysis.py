import util_bosch as ub
import os
import glob
import pandas as pd

output_dir = ub.output_dir

f_list = glob.glob(os.path.join(ub.output_dir, 'run_info*24of24*'))

print '\n'.join(f_list)

d_list = []
for f in f_list:
    with open(f, 'r') as fp:
        print 'reading '+os.path.basename(f)
        if os.path.basename(f) == 'run_info_2016-10-26 19:46:38_24of24_2016-10-27 07:51:25.txt':
            print 'skipping'
            continue
        d = eval(fp.read())
        d['fname'] = os.path.basename(f)
        d_list.append(d)

df = pd.DataFrame(d_list)

df.to_csv(os.path.join(ub.code_dir, 'overall_results.csv'))


