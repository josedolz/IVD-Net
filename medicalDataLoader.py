from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_fat_path = os.path.join(root, 'train', 'Fat')
        train_inn_path = os.path.join(root, 'train', 'Inn')
        train_opp_path = os.path.join(root, 'train', 'Opp')
        train_wat_path = os.path.join(root, 'train', 'Wat')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_opp.sort()
        images_wat.sort()
        labels.sort()

        for it_f,it_i,it_o,it_w, it_gt in zip(images_fat,images_inn,images_opp,images_wat,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
            
    elif mode == 'val':
        train_fat_path = os.path.join(root, 'val', 'Fat')
        train_inn_path = os.path.join(root, 'val', 'Inn')
        train_opp_path = os.path.join(root, 'val', 'Opp')
        train_wat_path = os.path.join(root, 'val', 'Wat')
        train_mask_path = os.path.join(root, 'val', 'GT')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_opp.sort()
        images_wat.sort()
        labels.sort()

        for it_f,it_i,it_o,it_w, it_gt in zip(images_fat,images_inn,images_opp,images_wat,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
    else:
        train_fat_path = os.path.join(root, 'test', 'Fat')
        train_inn_path = os.path.join(root, 'test', 'Inn')
        train_opp_path = os.path.join(root, 'test', 'Opp')
        train_wat_path = os.path.join(root, 'test', 'Wat')
        train_mask_path = os.path.join(root, 'test', 'GT')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_opp.sort()
        images_wat.sort()
        labels.sort()

        for it_f,it_i,it_o,it_w, it_gt in zip(images_fat,images_inn,images_opp,images_wat,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    # If we want to augment the data online
    def augment(self, img, mask):
        angle = random() * 10 - 20
        img_f = img_f.rotate(angle)
        img_i = img_i.rotate(angle)
        img_o = img_o.rotate(angle)
        img_w = img_w.rotate(angle)
        mask = mask.rotate(mask)
        return img_f,img_i,img_o,img_w,mask

    def __getitem__(self, index):
        fat_path, inn_path,opp_path,wat_path,mask_path = self.imgs[index]

        img_f = Image.open(fat_path)#.convert('L')
        img_i = Image.open(inn_path)#.convert('L')
        img_o = Image.open(opp_path)#.convert('L')
        img_w = Image.open(wat_path)#.convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img_f = self.transform(img_f)
            img_i = self.transform(img_i)
            img_o = self.transform(img_o)
            img_w = self.transform(img_w)
            mask = self.mask_transform(mask)

        return [img_f,img_i,img_o,img_w, mask, fat_path]
