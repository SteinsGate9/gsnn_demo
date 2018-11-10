# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import h5py
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import torchvision.datasets as dset
import torchvision.transforms as transforms

class vgDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(cocoDataLoader, self).__init__(*args, **kwargs)

class cocoDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(cocoDataLoader, self).__init__(*args, **kwargs)

class vgDataSet(object):
    pass

class cocoDataSet(object):
    """
    Load coco images for Gsnn
    """
    def __init__(self, data_path, data_type):
        # initialize COCO api for instance annotations
        self.path = os.path.join(data_path,data_type)
        self.f = h5py.File(self.path + '.h5', 'r')
        # put into memory
        self.cats = self.f['cats']
        self.image_h = self.f[data_type + '_images']
        self.name_h = self.f[data_type + '_image_names']
        self.shape_h = self.f[data_type + '_image_shapes']
        self.label_h = self.f[data_type + '_labels']

    def get_cats(self):
        return self.cats

    def __getitem__(self, index):
        # show random images to test
        img = self.image_h[index]
        label = self.label_h[index]
        shape = self.shape_h[index]
        name = self.name_h[index]
        return img, label, shape, name

    def __len__(self):
        return len(self.name_h)


dataDir=''
dataType='train'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
gg = cocoDataSet(dataDir,dataType)
print(len(gg))
c = 0
from tqdm import tqdm
pbar = tqdm(range(len(gg)))
for a in pbar:
    print(gg[a][0])
    print(gg[a][1])
    print(gg[a][2])
    print(gg[a][3])
print(c)





# pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# dataDir='L:/coco'
# dataType='train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco=COCO(annFile)
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds([1234]))
# nms=[cat['name'] for cat in cats]
# print(len(nms))
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# # get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# imgIds = coco.getImgIds(catIds=catIds )
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# # load and display image
# I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
#
# # initialize COCO api for person keypoints annotations
# annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)
#
# # load and display keypoints annotations
# plt.imshow(I); plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)
#
#
#
