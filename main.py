# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""

import os
import sys

sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from data.image.cocoDataSet import vgDataSet
from data.image.cocoDataSet import vgDataLoader
from data.graph.vgDataSet import GraphLoader
from models.gsnn import GSNN

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--projectpath',type=str, default="NUSWID",help='which dataset to use')
parser.add_argument('--dataset',type=str, default="NUSWID",help='which dataset to use')
parser.add_argument('--datapath',type=str, default="L:/vg/VG_100K_2",help='where to place your data')
parser.add_argument('--vggpath',type=str, default="vgg/vgg16-397923af.pth",help='where to place your data')
parser.add_argument('--datapath2',type=str, default="L:/vg/VG_100K_2",help='where to place your data')
parser.add_argument('--label_dir',type=str, default="label_dict.json",help='where to place your data')
parser.add_argument('--label2_dir',type=str, default="label_dict.json",help='where to place your data')
parser.add_argument('--concat_dir',type=str, default="concat.json",help='where to place your data')
parser.add_argument('--anno_dir',type=str, default="annotation.json",help='where to place your data')
parser.add_argument('--graph_dir',type=str, default="annotation.json",help='where to place your data')

# question settings
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers_num', type=int, help='number of data loading workers', default=2)

# model settings
parser.add_argument('--vg_objects', type=int, default=200, help='vg objects for labels')
parser.add_argument('--vg_attributes', type=int, default=100, help='vg attributes for labels')
parser.add_argument('--coco_cats', type=int, default=23, help='coco cats for labels')
parser.add_argument('--label_num', type=int, default=80, help='coco cats for labels')
parser.add_argument('--label_len', type=int, default=323, help='coco cats for labels')
parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
parser.add_argument('--state_dim', type=int, default=5, help='GSNN hidden state size')
parser.add_argument('--edge_type_num', type=int, default=2, help='GSNN edge type')
parser.add_argument('--node_num', type=int, default=1278, help='GSNN hidden state size')
parser.add_argument('--annotation_dim', type=int, default=1, help='GSNN annotation state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--importance_factor', type=float, default=0.3, help='importance factor of gsnn')
parser.add_argument('--expand_num', type=int, default=5, help='expand the net per step')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='threshold')

# training settings
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='training weight decay')
parser.add_argument('--lr_decay_step', type=float, default=10, help='training weight decay')
parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--method_pipeline', type=str, default='SGD', help='training method')
parser.add_argument('--method_gsnn', type=str, default='Adam', help='training method')
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--penalty', type=int, default=2, help='L2 penalty')
parser.add_argument('--verbose', type=bool, default=True, help='print training info or not')
parser.add_argument('--manual_seed', type=int, help='manual seed')
opt = parser.parse_args()
opt.cuda = False
opt.project_path = os.getcwd()
opt.dataset = 'vgGenome'
opt.datapath =  '/home/nesa320/huangshicheng/gitforwork/my_gsnn/data/image/VG_100K'
opt.label_dir = opt.project_path + '/data/image/labels_dict.json'
opt.label2_dir = opt.project_path + '/data/image/labels.json'
opt.concat_dir = opt.project_path + '/data/image/concat_dict.json'
opt.anno_dir = opt.project_path + '/data/image/annotation_dict_v3.json'
opt.graph_dir = opt.project_path + '/data/graph/filtered_graph_v3.json'

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)


from models.gsnn import PipeLine
def main(opt):
    # set up data
    train_dataset = vgDataSet(image_dir=opt.datapath,
                              label_dir=opt.label_dir,
                              concat_dir=opt.concat_dir,
                              anno_dir=opt.anno_dir)
    train_dataloader = vgDataLoader(train_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=True, 
                                      num_workers=opt.workers_num)
    test_dataset = vgDataSet(image_dir=opt.datapath,
                              label_dir=opt.label_dir,
                              concat_dir=opt.concat_dir,
                              anno_dir=opt.anno_dir)
    test_dataloader = vgDataLoader(test_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=opt.workers_num)

    # set up model and train
    pipeline = PipeLine(opt=opt, dataloader=train_dataloader)
    pipeline._train()

    return







if __name__ == '__main__':
    main(opt)