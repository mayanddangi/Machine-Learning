import argparse
import os
from best import best_classifier_multi_class, best_classifier_two_class
import numpy as np
import pandas as pd
import sys
import traceback
    
    
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='../data/', help='path to data file')
parser.add_argument('--entry_number', type=str, default='20XXYYYZZZZ', help='entry number')
parser.add_argument('--logs', type=str, default='../logs/', help='path to logs file')
parser.add_argument('--result', type=str, default='../result/', help='path to results')
parser.add_argument('--split', type=str, default='test', help='testing split')
args = parser.parse_args()

sys.stdout = open(os.path.join(args.logs + args.entry_number + '.txt'), 'w')
sys.stderr = sys.stdout
# import pdb; pdb.set_trace()
# labels_two_class = pd.read_csv(os.path.join(args.data + 'labels_two_class.csv')).loc[:,'y'].to_numpy()
# labels_multi_class = pd.read_csv(os.path.join(args.data + 'labels_multi_class.csv')).loc[:,'y'].to_numpy()

print('Binary Classification\n')
try:        
    two_class = best_classifier_two_class()
    two_class.fit(os.path.join(args.data + 'train_two_class.csv'))
    preds_two_class = two_class.predict(os.path.join(args.data + f'{args.split}_two_class.csv'))
except Exception as e:
    print(traceback.format_exc())
    preds_two_class = []
print('Multiclass Classification\n')
try:
    multi_class = best_classifier_multi_class()
    multi_class.fit(os.path.join(args.data + 'train_multi_class.csv'))
    preds_multi_class = multi_class.predict(os.path.join(args.data + f'{args.split}_multi_class.csv'))
except Exception as e:
    print(traceback.format_exc())
    preds_multi_class = []
preds_multi_class = preds_multi_class.astype(int).flatten() if len(preds_multi_class) > 0 else []
preds_two_class = preds_two_class.astype(int).flatten() if len(preds_two_class) > 0 else []
os.makedirs(os.path.join(args.result,'two_class'), exist_ok=True)
os.makedirs(os.path.join(args.result,'multi_class'), exist_ok=True)

with open(os.path.join(args.result,'two_class',args.entry_number +'.txt'), 'w') as f:
    f.write('\n'.join([str(x) for x in preds_two_class]))

with open(os.path.join(args.result,'multi_class',args.entry_number +'.txt'), 'w') as f:
    f.write('\n'.join([str(x) for x in preds_multi_class]))