# -*- coding: UTF-8 -*-
'''
@Project ：MyModel 
@File    ：my_hyper_pytorch.py
@Author  ：xiaoliusheng
@Date    ：2023/9/26/026 21:26 
'''

import numpy as np

import torch
from  torch.utils.data.dataset import Dataset

class HyperData_classifier(Dataset):
    def __init__(self,HR_data,LR_data,LR_gt, transfor):
        self.HR_data = HR_data
        self.LR_data = LR_data
        self.transformer = transfor
        self.labels = []
        for n in LR_gt: self.labels += [int(n)]

    def __getitem__(self, index):
        HR_img = torch.from_numpy(np.asarray(self.HR_data[index, :, :, :]))
        LR_img = torch.from_numpy(np.asarray(self.LR_data[index, :, :, :]))
        label = self.labels[index]
        return HR_img, LR_img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels

class HyperData_SR(Dataset):
    def __init__(self, hr_image, lr_image,labels):
        self.hr_image = hr_image.astype(np.float32)  # 高分辨率图像  c,h,w
        self.lr_image = lr_image.astype(np.float32)  # 低分辨率图像
        self.labels = labels.astype(np.int64)  # 标签图像 h,w


    def __len__(self):
        # 计算可以从HR图像中提取多少个图像块
        return len(self.labels)


    def __getitem__(self, idx):
        hr_patch = self.hr_image[idx,:,:,:]
        lr_patch = self.lr_image[idx,:,:,:]
        label_patch = self.labels[idx,:,:]
        return torch.tensor(hr_patch), torch.tensor(lr_patch),torch.tensor(label_patch)

class HyperData_SR_1(Dataset):
    def __init__(self, hr_image, lr_image,labels, hr_patch_size, lr_patch_size):
        self.hr_image = hr_image.astype(np.float32)  # 高分辨率图像  c,h,w
        self.lr_image = lr_image.astype(np.float32)  # 低分辨率图像
        self.labels = labels.astype('uint8')  # 标签图像 h,w
        self.hr_patch_size = hr_patch_size  # 高分辨率图像块大小
        self.lr_patch_size = lr_patch_size  # 低分辨率图像块大小

        self.n_patches_x = self.hr_image.shape[1] // self.hr_patch_size[0]
        self.n_patches_y = self.hr_image.shape[2] // self.hr_patch_size[1]
        self.num_patch = self.n_patches_x * self.n_patches_y

        # self.hr_patch = np.zeros((self.num_patch,hr_image.shape[0],hr_patch_size[0],hr_patch_size[1]),dtype='float32')
        # self.lr_patch = np.zeros((self.num_patch,lr_image.shape[0],lr_patch_size[0],lr_patch_size[1]),dtype='float32')



    def __len__(self):
        # 计算可以从HR图像中提取多少个图像块
        return self.num_patch

    def gt_len(self):
        # 计算可以从HR图像中提取多少个图像块
        return self.hr_patch_size[0]*self.hr_patch_size[1]

    def __getitem__(self, idx):
        # 计算图像块的坐标
        patch_x = idx % self.n_patches_x
        patch_y = idx //self.n_patches_x

        ##!!!!!!
        # 提取HR和LR图像块
        hr_patch = self.hr_image[:,
            patch_x*self.hr_patch_size[0]:(patch_x+1)*self.hr_patch_size[0],
            patch_y*self.hr_patch_size[1]:(patch_y+1)*self.hr_patch_size[1],
        ]
        lr_patch = self.lr_image[:,
            patch_x*self.lr_patch_size[0]:(patch_x+1)*self.lr_patch_size[0],
            patch_y*self.lr_patch_size[1]:(patch_y+1)*self.lr_patch_size[1],
        ]

        # 提取对应的标签块
        label_patch = self.labels[
                      patch_x * self.hr_patch_size[0]:(patch_x + 1) * self.hr_patch_size[0],
                      patch_y * self.hr_patch_size[1]:(patch_y + 1) * self.hr_patch_size[1]
                      ]

        # # 如果标签块小于hr_patch_size，则进行填充
        # if label_patch.shape[0] < self.hr_patch_size[0] or label_patch.shape[1] < self.hr_patch_size[1]:
        #     label_padded = np.zeros(self.hr_patch_size, dtype=label_patch.dtype)
        #     label_padded[:label_patch.shape[0], :label_patch.shape[1]] = label_patch
        #     label_patch = label_padded

        return torch.tensor(hr_patch), torch.tensor(lr_patch),torch.tensor(label_patch)
