# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import torchvision.datasets as dset
import torchvision.transforms as transforms

class vgDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(vgDataLoader, self).__init__(*args, **kwargs)

class cocoDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(cocoDataLoader, self).__init__(*args, **kwargs)

import json
class vgDataSet(object):
    """
       Load vg images for Gsnn
       """
    def __len__(self) -> int:
        return len(self.image_dir)

    def __init__(self, image_dir: str, label_dir:str, concat_dir:str, anno_dir:str):
        super().__init__()
        self.image_dir = [os.path.join(image_dir,f) for f in os.listdir(image_dir)]
        self.image_name = [i.split("/")[-1] for i in self.image_dir]
        #self.label_data = json.loads(label_dir) ## 316
        self.label_data = dict.fromkeys(self.image_name, np.zeros(316))
        #self.concat_data = json.loads(concat_dir) ## 80
        self.concat_data =dict.fromkeys(self.image_name, np.zeros(80))
        #self.annotation_data = json.loads(anno_dir) ## 919
        self.annotation_data = dict.fromkeys(self.image_name, np.zeros(919))
        self.transform = transforms.Compose(transforms = [
            transforms.Scale(size=224),
            transforms.ToTensor()
        ])

    def __getitem__(self, index: int):
        # deal with image
        img_path = self.image_dir[index]
        img_name = self.image_name[index]
        with Image.open(img_path) as img:
            image = img.convert('RGB')

        # deal with others
        label = self.label_data[img_name]
        concat = self.concat_data[img_name]
        anno = self.annotation_data[img_name]

        # turn to torch
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label).requires_grad_()
        concat = torch.FloatTensor(concat).requires_grad_()
        anno = torch.FloatTensor(anno).requires_grad_()

        # cuda
        cuda = True
        if cuda:
            image = image.cuda()
            concat = concat.cuda()
            label = label.cuda()
            anno = anno.cuda()

        return image, label, concat, anno






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

if __name__ == "main":
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
