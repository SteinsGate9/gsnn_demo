import sys
import os
sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import DataGenerator


parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--dataset',type=str, default="NUSWID",help='which dataset to use')
parser.add_argument('--datapath',type=str, default="L:/",help='where to place your data')

# question settings
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

# model settings
parser.add_argument('--annotation_dim', type=int, default=1, help='dim for annotation')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')

# training settings
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manual_seed', type=int, help='manual seed')
opt = parser.parse_args()

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

opt.dataroot = 'babi_data/processed_1/train/%d_graphs.txt' % opt.task_id


def main(opt):
    data_generator = DataGenerator(opt)
    return







if __name__ == '__main__':
    main(opt)