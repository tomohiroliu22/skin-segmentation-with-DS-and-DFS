import argparse
import os
from train_DFS_w_DS import train_dfs_w_ds, test_dfs_w_ds
from train_DFS import train_dfs, test_dfs
from train_DS import train_ds, test_ds
from train_NONE import train_none, test_none

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to image and label dataset')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--phase', type=str, default='train', help='[train | test]')
parser.add_argument('--model', type=str, default='DFS_w_DS', help='chooses which model to use. [DFS_w_DS | DFS | DS | NONE]')
parser.add_argument('--fold', type=int, default=0, help='chooses which fold to use')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--load_model', type=bool, default=False, help='load pre-trainined model to fine-tune')
parser.add_argument('--modelpath', type=str, default="your model path", help='the model path to load')
parser.add_argument('--lr', type=float, default="0.001", help='learning rate')
parser.add_argument('--step', type=int, default="10", help='step size of scheduler')
parser.add_argument('--epoch', type=int, default="25", help='number of epochs to train')
opts = parser.parse_args()

if(opts.phase=='train'):
    if(opts.model == 'DFS_w_DS'):   
        train_dfs_w_ds(opts)
    elif(opts.model == 'DFS'):  
        train_dfs(opts)
    elif(opts.model == 'DS'):  
        train_ds(opts)
    else:
        train_none(opts)
if(opts.phase=='test'):
    opts.load_model = True
    if(opts.model == 'DFS_w_DS'):   
        test_dfs_w_ds(opts)
    elif(opts.model == 'DFS'):  
        test_dfs(opts)
    elif(opts.model == 'DS'):  
        test_ds(opts)
    else:
        test_none(opts)