# -*- coding: UTF-8 -*-
'''
@Project ：MyModel 
@File    ：my_dataloder.py
@Author  ：xiaoliusheng
@Date    ：2023/9/26/026 21:36 
'''
import os
import random

import torchvision
from math import ceil

import scipy.io as sio
import scipy.ndimage
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

from tools.my_hyper_pytorch import *
from torch.utils.data import random_split, DataLoader
import time
from datetime import datetime
import logging
from tools.show_maps import *

# 切块右边补0 下边补0
def loadData(data_path, name, num_components=None):
    if name == "PC":  # pavia center
        hr_data = sio.loadmat(os.path.join(data_path, 'Pavia Center scene', 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia Center scene', 'Pavia_gt.mat'))['pavia_gt']

    elif name == "PU":  # paviaU
        hr_data = sio.loadmat(os.path.join(data_path, 'Pavia University scene', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia University scene', 'PaviaU_gt.mat'))['paviaU_gt']

        # lr_data = sio.loadmat(os.path.join(data_path, 'paviaU_down_4X', 'PaviaU_down_4X.mat'))[
        #     'paviaU_down_4X']
        label_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees',
                       'Painted metal sheets', 'Bare Soil', 'Bitumen',
                       'Self-Blocking Bricks', 'Shadows']
    elif name == "SA":  # Salinas
        hr_data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected', 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_corrected', 'Salinas_gt.mat'))['salinas_gt']
    elif name == "IP":  # Indian_pines
        hr_data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected', 'Indian_pines_corrected.mat'))[
            'indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected', 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == "KSC":  # KSC
        hr_data = sio.loadmat(os.path.join(data_path, 'Kennedy Space Center (KSC)', 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'Kennedy Space Center (KSC)', 'KSC_gt.mat'))['KSC_gt']

    else:
        print('NO DATASET')
        exit()

    #lr_data 获取
    # 确保hr_data是一个PyTorch张量
    # 将 NumPy 数组转换为 float32 或 float64 类型
    hr_data_float = hr_data.astype(np.float32)
    hr_data_float = np.transpose(hr_data_float, (2, 0, 1))
    hr_data_float = torch.tensor(hr_data_float, dtype=torch.float32)

    # 如果hr_data是三维的 (channels, height, width)，需要增加一个batch维度
    if len(hr_data_float.shape) == 3:
        hr_data_float = hr_data_float.unsqueeze(0)  # 添加batch维度
    # 执行双三次下采样
    lr_data = F.interpolate(hr_data_float, scale_factor=0.25, mode='bicubic', align_corners=False)
    # 如果需要，移除batch维度
    lr_data = lr_data.squeeze(0)
    lr_data = lr_data.cpu().numpy()
    lr_data = np.transpose(lr_data, (1, 2, 0))


    # hr_data
    shapeor_hr = hr_data.shape
    hr_data = hr_data.reshape(-1, hr_data.shape[-1])
    # 指定降维操作
    if num_components != None:
        print("PCA successful!")
        hr_data = PCA(n_components=num_components).fit_transform(hr_data)
        shapeor_hr = np.array(shapeor_hr)
        shapeor_hr[-1] = num_components

    hr_data = MinMaxScaler().fit_transform(hr_data)
    # hr_data = StandardScaler().fit_transform(hr_data)
    hr_data = hr_data.reshape(shapeor_hr)

    # lr_data
    shapeor_lr = lr_data.shape
    lr_data = lr_data.reshape(-1, lr_data.shape[-1])
    # 指定降维操作
    if num_components != None:
        lr_data = PCA(n_components=num_components).fit_transform(lr_data)
        shapeor_lr = np.array(shapeor_lr)
        shapeor_lr[-1] = num_components
    lr_data = MinMaxScaler().fit_transform(lr_data)
    lr_data = lr_data.reshape(shapeor_lr)

    num_class = len(np.unique(labels)) - 1

    return hr_data, lr_data, labels, num_class


def load_hyper_sr(data_path, name, lr_spatial_size=5, up_scale=4, train_percent=0.6,train_num=50,
                  batch_size=32, components=None, rand_state=None, filename=None):
    # 配置日志记录器
    logging.basicConfig(filename=filename, level=logging.INFO)
    hr_data, lr_data, labels, num_classes = loadData(data_path, name, components)
    bands = hr_data.shape[-1]
    show_label(hr_data, labels, num_classes, os.path.join('/media/xd132/USER/XLS/TGRS/MTLSC-Diff_2.22/GtMap', name + '_gt.png'))
    # plt.close()
    print('{}, 总数据大小:{},{}, 总样本标签大小:{},类别数量：{}'.format(name, hr_data.shape, lr_data.shape,
                                                                       labels.shape,
                                                                       num_classes))
    logging.info('{}, 总数据大小:{},{}, 总样本标签大小:{},类别数量：{}'.format(name, hr_data.shape, lr_data.shape,
                                                                              labels.shape,
                                                                              num_classes))
    print('对lr和hr切图,作为SR的输入')

    hr_patch_size = (lr_spatial_size * up_scale, lr_spatial_size * up_scale)
    lr_patch_size = (lr_spatial_size, lr_spatial_size)

    hr_data = get_patches_data(np.transpose(hr_data, (2, 0, 1)).astype("float32"), hr_patch_size)
    lr_data = get_patches_data(np.transpose(lr_data, (2, 0, 1)).astype("float32"), lr_patch_size)
    labels = get_patches_gt(labels, hr_patch_size)

    print('全图切块个数,hr_data, lr_data, labels',len(hr_data), len(lr_data), len(labels))

    train_size = int(len(hr_data) * train_percent)  # 假设训练集占40%
    hr_data = np.array(hr_data)
    lr_data = np.array(lr_data)
    labels = np.array(labels)

    train_hr_data, train_lr_data, train_label = random_unison_train(hr_data, lr_data, labels, train_size, rand_state)
    print('训练集切块个数,hr_data, lr_data, labels', len(train_hr_data), len(train_lr_data), len(train_label))

    dataset = HyperData_SR(hr_data,lr_data, labels)
    train_dataset = HyperData_SR(train_hr_data, train_lr_data, train_label)
    del hr_data, lr_data, labels, train_hr_data, train_lr_data, train_label

    kwargs = {'num_workers': 1, 'pin_memory': True}
    # # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_classes, bands


def load_hyper_sr_1(data_path, name, lr_spatial_size=5, up_scale=4, train_percent=0.6,train_num=50,
                  batch_size=32, components=None, rand_state=None, filename=None):
    # 配置日志记录器
    logging.basicConfig(filename=filename, level=logging.INFO)
    hr_data, lr_data, labels, num_classes = loadData(data_path, name, components)
    bands = hr_data.shape[-1]
    # print(type(hr_data), type(labels), type(num_classes))
    #
    # print(hr_data.shape, lr_data.shape, labels.shape, num_classes)

    # plt.imshow(labels)
    # # colors.ListedColormap(color_matrix))
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # print( type(hr_data), hr_data.shape)
    # print( type(labels), labels.shape)
    print(labels.dtype)
    show_label(hr_data, labels, num_classes, os.path.join('/media/xd132/USER/XLS/TGRS/EMLSC-Diff_1.8/save_img', name + '_gt.png'))
    # plt.close()
    print('{}, 总数据大小:{},{}, 总样本标签大小:{},类别数量：{}'.format(name, hr_data.shape, lr_data.shape,
                                                                       labels.shape,
                                                                       num_classes))
    logging.info('{}, 总数据大小:{},{}, 总样本标签大小:{},类别数量：{}'.format(name, hr_data.shape, lr_data.shape,
                                                                              labels.shape,
                                                                              num_classes))
    print('对lr和hr切图,作为SR的输入')

    hr_patch_size = (lr_spatial_size * up_scale, lr_spatial_size * up_scale)
    lr_patch_size = (lr_spatial_size, lr_spatial_size)

    hr_data = get_patches_data(np.transpose(hr_data, (2, 0, 1)).astype("float32"), hr_patch_size)
    lr_data = get_patches_data(np.transpose(lr_data, (2, 0, 1)).astype("float32"), lr_patch_size)
    labels = get_patches_gt(labels, hr_patch_size)

    # target_gt = np.zeros((1096,715)).astype('uint8')
    # print('target_gt',type(target_gt),target_gt.shape)
    # recon_gt=get_all_gt(target_gt, hr_patch_size, labels)
    # print(recon_gt.dtype)
    # print(type(recon_gt), recon_gt.shape)
    # show_label(target_gt, recon_gt, num_classes, os.path.join('/media/xd132/USER/XLS/TGRS/EMLSC-Diff_1.8/save_img', name + '_gt_recon_160.png'))
    # plt.close()

    print('全图切块个数,hr_data, lr_data, labels',len(hr_data), len(lr_data), len(labels))

    train_size = int(len(hr_data) * train_percent)  # 假设训练集占40%
    hr_data = np.array(hr_data)
    lr_data = np.array(lr_data)
    labels = np.array(labels)
    print(hr_data.shape, lr_data.shape, labels.shape)
    train_hr_data, train_lr_data,train_label = random_unison_train(hr_data, lr_data, labels,  train_size, rand_state)
    print('训练集切块个数,hr_data, lr_data, labels', len(train_hr_data), len(train_lr_data), len(train_label))

    train_label_progress , train_total_samples, class_samples_count = data_partition(samples_type='same_num',
                                        class_count=num_classes,
                                        gt=train_label,
                                        train_ratio=0.1,
                                        val_ratio=0.1,
                                        N=train_num,
                                        margin=1)
    print("train_label-{} train_total_samples-{} class_samples_count-{}".format(train_label_progress.shape, train_total_samples,
                                                                                class_samples_count))

    dataset = HyperData_SR(hr_data,lr_data, labels)
    train_dataset = HyperData_SR(train_hr_data, train_lr_data, train_label_progress)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    # # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    #
    # # for idx, (hr_patch, lr_patch, gt) in enumerate(train_loader):
    # #     print(hr_patch.shape)
    # #     print(lr_patch.shape)
    # #     print(gt.shape)
    # #
    # #     # randn3 = np.random.randint(0,9)
    # #     # print(randn3)
    # #     randn3 = 0
    # #     HR_ = hr_patch[randn3]
    # #     LR_ = lr_patch[randn3]
    # #     gt_ = gt[randn3]
    # #     plt.figure(figsize=(15, 10))
    # #     plt.subplot(1, 3, 1)
    # #     plt.axis("off")
    # #     plt.title("LR", fontsize=20)
    # #     plt.imshow(np.transpose(torchvision.utils.make_grid(LR_,
    # #                                                         nrow=2, padding=1, normalize=True).cpu(),
    # #                             (1, 2, 0))[:, :, [22, 10, 3]])
    # #
    # #     plt.subplot(1, 3, 2)
    # #     plt.axis("off")
    # #     plt.title("HR", fontsize=20)
    # #     plt.imshow(np.transpose(torchvision.utils.make_grid(HR_,
    # #                                                         nrow=2, padding=1, normalize=True).cpu(),
    # #                             (1, 2, 0))[:, :, [22, 10, 3]])
    # #     plt.subplot(1, 3, 3)
    # #     plt.axis("off")
    # #     plt.title("GT", fontsize=20)
    # #     plt.imshow(gt_)
    # #
    # #     # 添加超级标题
    # #     plt.suptitle("idx-{}  TEST-LR VS HR".format(idx), fontsize=25)
    # #     plt.show()
    # #     plt.close()
    #
    #     # print(HR_.shape, gt_.shape)
    #     # show_label(np.transpose(torchvision.utils.make_grid(HR_,normalize=True).cpu(),
    #     #                         (1, 2, 0)), gt_, num_classes, os.path.join('/media/xd132/USER/XLS/TGRS/EMLSC-Diff_12.1/sava_img', name + '_gt-{}.png'.format(randn3)))

    return train_loader, test_loader, num_classes, bands

def get_patches_data(data, patch_size):
    C, H, W = data.shape
    h, w = patch_size

    # 计算每个方向上所需的填充量
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w

    # 将所有填充都放在右侧和底部
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    # 使用对称进行填充
    padded_data = np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')

    # 切割填充后的数据
    patches = []
    C, H, W = padded_data.shape
    for i in range(0, H - h + 1, h):
        for j in range(0, W - w + 1, w):
            patch = padded_data[:, i:i + h, j:j + w]
            patches.append(patch)
    return patches

def get_patches_gt(gt, patch_size):
    H, W = gt.shape
    h, w = patch_size

    # 计算每个方向上所需的填充量
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w

    # 将所有填充都放在右侧和底部
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    # 使用0进行填充
    padded_gt = np.pad(gt, ( (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant').astype('long')

    # 切割填充后的数据
    patches = []
    H, W = padded_gt.shape
    for i in range(0, H - h + 1, h):
        for j in range(0, W - w + 1, w):
            patch = padded_gt[i:i + h, j:j + w]
            patches.append(patch)
    return patches


def get_all_gt(target_gt, patch_size, gt):
    target_H, target_W = target_gt.shape
    h, w = patch_size

    # 计算每个方向上所需的填充量
    pad_h = (h - target_H % h) % h
    pad_w = (w - target_W % w) % w

    # 将所有填充都放在右侧和底部
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    # 使用0进行填充
    padded_gt = np.pad(target_gt, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant').astype('long')

    H, W = padded_gt.shape
    # 使用嵌套循环遍历原始图像，并将小patch放回相应位置
    patch_index = 0
    for i in range(0, H - h + 1, h):
        for j in range(0, W - w + 1, w):
            patch = gt[patch_index]
            # print(patch.shape)
            padded_gt[i:i + h, j:j + w] = patch.reshape(h, w)
            patch_index += 1

    # 计算去掉填充后的图像的高度和宽度
    new_height = H - pad_top - pad_bottom
    new_width = W - pad_left - pad_right

    # 使用切片操作去掉填充部分
    original_image = padded_gt[pad_top:pad_top + new_height, pad_left:pad_left + new_width]

    return original_image
def data_partition(samples_type, class_count, gt, train_ratio, val_ratio, N=10,margin=3):
    train_data_index = []
    test_data_index = []
    val_data_index = []
    class_samples_count = {}  # 添加字典来存储每个类别的样本数量

    # 转换gt为numpy数组（如果它是一个Tensor）
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().detach().numpy()
    # gt_reshape = np.reshape(gt, [-1])

    # 为每个样本和每个类别生成索引
    if samples_type == 'ratio':
        train_data_index = set(train_data_index)
        for sample in range(gt.shape[0]):  # 遍历每个样本
            for cls in range(1, class_count + 1):  # 遍历每个类别
                cls_indices = np.transpose(np.nonzero(gt[sample] == cls))
                # 过滤掉靠近边界的样本
                cls_indices = [idx for idx in cls_indices if
                               idx[0] >= margin and idx[1] >= margin and idx[0] < gt.shape[1] - margin and idx[1] <
                               gt.shape[2] - margin]
                cls_sample_count = int(np.ceil(len(cls_indices) * train_ratio))
                class_samples_count[cls] = class_samples_count.get(cls, 0) + cls_sample_count
                if cls_sample_count > 0:
                    selected_indices = random.sample(list(cls_indices), cls_sample_count)
                    train_data_index.update({(sample, idx[0], idx[1]) for idx in selected_indices})

            # 转换为set以便快速操作
            # train_data_index = set(train_data_index)
            all_data_index = set(
                [(i, j, k) for i in range(gt.shape[0]) for j in range(gt.shape[1]) for k in range(gt.shape[2])])

            # 标记背景像素
            background_idx = set(
                [(sample, x, y) for sample in range(gt.shape[0]) for x, y in np.transpose(np.nonzero(gt[sample] == 0))])

            # 生成测试和验证集索引
            test_data_index = all_data_index - train_data_index - background_idx
            val_data_count = int(val_ratio * len(test_data_index))
            val_data_index = set(random.sample(test_data_index, val_data_count))
            test_data_index -= val_data_index

        # 将索引转换为列表
        train_data_index = list(train_data_index)
        test_data_index = list(test_data_index)
        val_data_index = list(val_data_index)

        # 生成训练、测试和验证标签图
        train_total_samples = sum(class_samples_count.values())
        train_label = np.zeros_like(gt)
        for idx in train_data_index:
            train_label[idx[0], idx[1], idx[2]] = gt[idx[0], idx[1], idx[2]]

        test_label = np.zeros_like(gt)
        for idx in test_data_index:
            test_label[idx[0], idx[1], idx[2]] = gt[idx[0], idx[1], idx[2]]

        val_label = np.zeros_like(gt)
        for idx in val_data_index:
            val_label[idx[0], idx[1], idx[2]] = gt[idx[0], idx[1], idx[2]]

    if samples_type == 'same_num':  # 每个类别采样相同数量的样本
        train_data_index = set(train_data_index)
        for cls in range(1, class_count + 1):  # 遍历每个类别
            cls_indices = np.transpose(np.nonzero(gt == cls))  # 获取当前类别的所有索引
            # 过滤掉靠近边界的样本
            cls_indices = [idx for idx in cls_indices if
                           idx[0] >= margin and idx[1] >= margin and idx[0] < gt.shape[1] - margin and idx[1] <
                           gt.shape[2] - margin]
            cls_sample_count = min(len(cls_indices), N)  # 确定每个类别的样本数
            class_samples_count[cls] = cls_sample_count
            if cls_sample_count > 0:
                selected_indices = random.sample(list(cls_indices), cls_sample_count)
                train_data_index.update({(idx[0], idx[1], idx[2]) for idx in selected_indices})

        # 将索引转换为列表
        train_data_index = list(train_data_index)

        train_total_samples = sum(class_samples_count.values())
        # 生成训练、测试和验证标签图
        train_label = np.zeros_like(gt)
        for idx in train_data_index:
            train_label[idx[0], idx[1], idx[2]] = gt[idx[0], idx[1], idx[2]]

    return train_label, train_total_samples, class_samples_count

# padding
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 创建像素立方体
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    num_labels = np.count_nonzero(y[:, :])  # 标签样本总数, 非0数
    # print('标签样本总数(除去非0数)',num_labels)
    if windowSize % 2 == 0:
        margin = int(windowSize / 2)
    else:
        margin = int((windowSize - 1) / 2)

    zeroPaddedX = padWithZeros(X, margin=margin)
    # print('padding之后数据大小', zeroPaddedX.shape)
    # split patches
    patchIndex = 0
    if removeZeroLabels == True:
        patchesData = np.zeros((num_labels, windowSize, windowSize, X.shape[2]), dtype='float32')
        patchesLabels = np.zeros(num_labels)
        # print('patch数据总的格式', patchesData.shape, 'patch标签总数量', len(patchesLabels))

        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if y[r - margin, c - margin] > 0:
                    if windowSize % 2 == 0:
                        patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                    else:
                        patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]

                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    patchIndex = patchIndex + 1
                    # print(r, c, patchIndex)

    if removeZeroLabels == False:  # 表示使用全部像素
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype="float32")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if windowSize % 2 == 0:
                    patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                else:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1
    patchesLabels -= 1

    return patchesData, patchesLabels.astype("int")


# 创建像素立方体
def my_createImageCubes(X, y, windowSize=5, removeZeroLabels=True, up_scale=4):  # up_scale下上采因子，
    num_labels = np.count_nonzero(y[:, :])  # 标签样本总数, 非0数
    # print('标签样本总数(除去非0数)', num_labels)
    if windowSize % 2 == 0:
        margin = int(windowSize / 2)
    else:
        margin = int((windowSize - 1) / 2)

    zeroPaddedX = padWithZeros(X, margin=margin)
    # print('padding之后数据大小', zeroPaddedX.shape)
    # split patches
    patchIndex = 0
    if removeZeroLabels == True:
        patchesData = np.zeros((num_labels, windowSize, windowSize, X.shape[2]), dtype='float32')
        patchesLabels = np.zeros(num_labels)
        # print('patch数据总的格式', patchesData.shape, 'patch标签总数量', len(patchesLabels))

        for r in range(margin, zeroPaddedX.shape[0] - margin, up_scale):
            for c in range(margin, zeroPaddedX.shape[1] - margin, up_scale):  # 隔一个 检索
                if 0 <= ((r - margin) // up_scale) < y.shape[0] and 0 <= ((c - margin) // up_scale) < y.shape[1]:
                    if y[(r - margin) // up_scale, (c - margin) // up_scale] > 0:
                        if windowSize % 2 == 0:
                            patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
                        else:
                            patch = zeroPaddedX[r - margin:r + margin + 1,
                                    c - margin:c + margin + 1]

                        patchesData[patchIndex, :, :, :] = patch
                        patchesLabels[patchIndex] = y[(r - margin) // up_scale, (c - margin) // up_scale]
                        patchIndex = patchIndex + 1

    if removeZeroLabels == False:  # 表示使用全部像素
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype="float32")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        for r in range(margin, zeroPaddedX.shape[0] - margin, up_scale):
            for c in range(margin, zeroPaddedX.shape[1] - margin, up_scale):
                if windowSize % 2 == 0:
                    patch = zeroPaddedX[r - margin:r + up_scale + margin, c - margin:c + up_scale + margin]
                else:
                    patch = zeroPaddedX[r - margin:r + up_scale + margin + 1, c - margin:c + up_scale + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[(r - margin) // up_scale, (c - margin) // up_scale]
                patchIndex = patchIndex + 1
    patchesLabels -= 1

    return patchesData, patchesLabels.astype("int")


def my_split_data(hr_pixels, lr_pixels, labels, train_percent, rand_state=None):
    hr_train_set_size = []  # 存储每类地物训练样本数
    lr_train_set_size = []  # 存储每类地物训练样本数

    for cl in np.unique(labels):
        hr_pixels_cl = len(hr_pixels[labels == cl])  # 第i类地物样本总数
        lr_pixels_cl = len(lr_pixels[labels == cl])  # 第i类地物样本总数

        # pixels_cl = min(ceil(pixels_cl * 0.3), n_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        hr_pixels_cl = round(hr_pixels_cl * train_percent)  # ceil - 向上取整  round 四舍五入
        lr_pixels_cl = round(lr_pixels_cl * train_percent)  # ceil - 向上取整  round 四舍五入

        hr_train_set_size.append(hr_pixels_cl)  # 存储每类地物的样本数
        lr_train_set_size.append(lr_pixels_cl)  # 存储每类地物的样本数

        # print('第{}类训练集样本数量：{}'.format(cl + 1, pixels_cl))
    # print('每类训练样本数',train_set_size)
    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
    hr_tr_size = int(sum(hr_train_set_size))
    lr_tr_size = int(sum(lr_train_set_size))

    hr_te_size = int(sum(pixels_number)) - int(sum(hr_train_set_size))
    lr_te_size = int(sum(pixels_number)) - int(sum(lr_train_set_size))

    # print('训练集数量{}，测试集数量{}'.format(tr_size, te_size))
    hr_sizetr = np.array([hr_tr_size] + list(hr_pixels.shape)[1:])
    lr_sizetr = np.array([lr_tr_size] + list(lr_pixels.shape)[1:])

    hr_sizete = np.array([hr_te_size] + list(hr_pixels.shape)[1:])
    lr_sizete = np.array([lr_te_size] + list(lr_pixels.shape)[1:])

    hr_train_x = np.empty((hr_sizetr))
    lr_train_x = np.empty((lr_sizetr))

    train_y = np.empty((hr_tr_size), dtype=int)
    # lr_train_y = np.empty((lr_tr_size), dtype=int)

    hr_X_test = np.empty((hr_sizete))
    lr_X_test = np.empty((lr_sizete))

    y_test = np.empty((hr_te_size), dtype=int)
    # lr_y_test = np.empty((lr_te_size), dtype=int)

    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        hr_pixels_cl = hr_pixels[labels == cl]
        lr_pixels_cl = lr_pixels[labels == cl]

        labels_cl = labels[labels == cl]

        hr_pixels_cl, lr_pixels_cl, labels_cl = random_unison(hr_pixels_cl, lr_pixels_cl, labels_cl, rstate=rand_state)

        for cont, (a, b, c) in enumerate(zip(hr_pixels_cl, lr_pixels_cl, labels_cl)):
            if cont < hr_train_set_size[cl]:
                hr_train_x[trcont, :, :, :] = a
                lr_train_x[trcont, :, :, :] = b
                train_y[trcont] = c
                trcont += 1
            else:
                hr_X_test[tecont, :, :, :] = a
                lr_X_test[tecont, :, :, :] = b
                y_test[tecont] = c
                tecont += 1

    return hr_train_x, lr_train_x, train_y, hr_X_test, lr_X_test, y_test


def split_data(pixels, labels, train_percent, rand_state=None):
    train_set_size = []  # 存储每类地物训练样本数
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
        # pixels_cl = min(ceil(pixels_cl * 0.3), n_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        pixels_cl = round(pixels_cl * train_percent)  # ceil - 向上取整  round 四舍五入
        train_set_size.append(pixels_cl)  # 存储每类地物的样本数
        # print('第{}类训练集样本数量：{}'.format(cl + 1, pixels_cl))
    # print('每类训练样本数',train_set_size)
    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    # print('训练集数量{}，测试集数量{}'.format(tr_size, te_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])
    train_x = np.empty((sizetr))
    train_y = np.empty((tr_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont, :, :, :] = a
                train_y[trcont] = b
                trcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1
    # X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=train_percent, stratify=train_y,
    #                                                   random_state=rand_state)
    # 不使用val

    return train_x, train_y, X_test, y_test


def split_data_threshold_random(pixels, labels, n_samples, train_percent, rand_state=None):
    train_set_size = []  # 存储每类地物训练样本数
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
        # pixels_cl = min(ceil(pixels_cl * 0.3), n_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        if pixels_cl < n_samples:
            pixels_cl = ceil(pixels_cl * 0.8)
        else:
            pixels_cl = n_samples
        train_set_size.append(pixels_cl)  # 存储每类地物的样本数
        print('第{}类训练集样本数量：{}'.format(cl + 1, pixels_cl))
    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    print('训练集数量{}，测试集数量{}'.format(tr_size, te_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])
    train_x = np.empty((sizetr))
    train_y = np.empty((tr_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont, :, :, :] = a
                train_y[trcont] = b
                trcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1
    # X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=train_percent, stratify=train_y,
    #                                                   random_state=rand_state)
    # 不使用val

    return train_x, train_y, X_test, y_test


def split_data_percent(pixels, labels, train_samples, val_samples, rand_state=None):
    train_set_size = []  # 存储每类地物训练样本数
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
        train_pixels_cl = min(ceil(pixels_cl * 0.3), train_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        train_set_size.append(train_pixels_cl)  # 存储每类地物的训练样本数

    val_set_size = [ceil(i * val_samples) for i in train_set_size]  # 存储每类地物的验证样本数

    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数

    tr_size = int(sum(train_set_size))
    val_size = int(sum(val_set_size))
    te_size = int(sum(pixels_number)) - tr_size - val_size
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizeval = np.array([val_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])

    X_train = np.empty((sizetr))
    y_train = np.empty((tr_size), dtype=int)
    X_val = np.empty((sizeval))
    y_val = np.empty((val_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    valcont = 0;
    tecont = 0;

    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                X_train[trcont, :, :, :] = a
                y_train[trcont] = b
                trcont += 1
            elif cont < train_set_size[cl] + val_set_size[cl]:
                X_val[valcont, :, :, :] = a
                y_val[valcont] = b
                valcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1

    X_train, y_train = random_unison(X_train, y_train, rstate=rand_state)
    # X_test, y_test = random_unison(X_test, y_test, rstate=rand_state)
    X_val, y_val = random_unison(X_val, y_val, rstate=rand_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

def random_unison_class(a, b,c,d,  rstate=None):
    assert len(a) == len(b)==len(c)==len(d)  # 用于判断一个表达式，在表达式条件为false时触发异常。
    p = np.random.RandomState(seed=rstate).permutation(len(a))  # 随机生成长度为len(a)的序列
    return a[p], b[p], c[p], d[p]

def random_unison(a, b, c, rstate=None):
    assert len(a) == len(b) == len(c)  # 用于判断一个表达式，在表达式条件为false时触发异常。
    p = np.random.RandomState(seed=rstate).permutation(len(a))  # 随机生成长度为len(a)的序列
    return a[p], b[p], c[p]

# def random_unison_train(a, b, c,train_size, rstate=None):
#     assert len(a) == len(b) == len(c)  # 用于判断一个表达式，在表达式条件为false时触发异常。
#     p = np.random.RandomState(seed=rstate).permutation(train_size)  # 随机生成长度为len(a)的序列
#     return a[p], b[p], c[p]
def random_unison_train(a, b, c, train_size, rstate=None):
    assert len(a) == len(b) == len(c)  # 用于判断一个表达式，在表达式条件为false时触发异常。

    # 定义每个部分的大小
    front_size = train_size // 3
    middle_size = train_size // 3
    end_size = train_size - front_size - middle_size

    # 对每个部分进行随机采样
    rng = np.random.RandomState(seed=rstate)
    front_indices = rng.choice(len(a) // 3, front_size, replace=False)
    middle_indices = rng.choice(len(a) // 3, middle_size, replace=False) + len(a) // 3
    end_indices = rng.choice(len(a) - len(a) // 3 * 2, end_size, replace=False) + len(a) // 3 * 2

    # 合并三个部分的随机样本
    p = np.concatenate([front_indices, middle_indices, end_indices])
    return a[p], b[p], c[p]



def split_train_data_threshold_random(lrHS, data_2X,hrHS, labels, n_samples, train_percent, rand_state=None):
    assert len(lrHS) == len(data_2X) == len(hrHS) == len(labels)  # 用于判断一个表达式，在表达式条件为false时触发异常。
    train_set_size = []  # 存储每类地物训练样本数
    for cl in np.unique(labels):
        pixels_cl = len(lrHS[labels == cl])  # 第i类地物样本总数
        # pixels_cl = min(ceil(pixels_cl * 0.3), n_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        if pixels_cl < n_samples:
            pixels_cl = ceil(pixels_cl * 0.8)
        else:
            pixels_cl = n_samples
        train_set_size.append(pixels_cl)  # 存储每类地物的样本数
        print('第{}类训练集样本数量：{}'.format(cl + 1, pixels_cl))
    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number))
    print('训练集数量{}，测试集数量{}'.format(tr_size, te_size))
    sizetr = np.array([tr_size] + list(lrHS.shape)[1:])
    print('sizetr:', sizetr)
    train_lr,train_2x,train_pred = np.empty((sizetr)),np.empty((sizetr)),np.empty((sizetr))
    # print('train_lr:', train_lr.shape)
    train_y = np.empty((tr_size), dtype=int)
    trcont = 0
    for cl in np.unique(labels):
        lr_cl = lrHS[labels == cl]
        data_2X_cl = data_2X[labels == cl]
        hr_cl = hrHS[labels == cl]
        labels_cl = labels[labels == cl]
        lr_cl,data_2X_cl,hr_cl, labels_cl = random_unison_class(lr_cl,data_2X_cl,hr_cl, labels_cl, rstate=rand_state)
        for cont, (a, b,c,d) in enumerate(zip(lr_cl,data_2X_cl,hr_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_lr[trcont, :, :, :] = a
                train_2x[trcont, :, :, :] = b
                train_pred[trcont, :, :, :] = c
                train_y[trcont] = d
                trcont += 1
    # X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=train_percent, stratify=train_y,
    #                                                   random_state=rand_state)
    # 不使用val
    train_lr_tensor = torch.from_numpy(train_lr)
    train_2x_tensor = torch.from_numpy(train_2x)
    train_pred_tensor = torch.from_numpy(train_pred)
    train_y_tensor = torch.from_numpy(train_y)
    train_y_tensor=train_y_tensor.long()
    return train_lr_tensor,train_2x_tensor,train_pred_tensor, train_y_tensor



if __name__ == '__main__':
    data_path = '/media/xd132/USER/XLS/data'
    name = 'PU'
    lr_spatial_size = 28
    up_scale = 4
    train_percent = 0.6
    train_num = 50
    batch_size_sr = 2
    batch_size_class = 128
    seed = 0
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if cuda else "cpu")
    filename = 'test_LoadData'
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    print(time_str)
    train_loader, test_loader, num_classes, bands = load_hyper_sr(data_path, name, lr_spatial_size=lr_spatial_size,
                                                                  up_scale=up_scale,
                                                                  train_percent=train_percent,train_num = train_num,
                                                                  batch_size=batch_size_sr, components=None,
                                                                  rand_state=seed, filename=filename)



    a, b, c = next(iter(train_loader))
    print("a:{},b:{},c:{}".format(a.shape, b.shape, c.shape))

    # 测试
    sample = 0
    for step, [hrHS, lrHS, gt] in enumerate(test_loader):
        gt_numpy = gt.numpy()
        non_zero_count = np.count_nonzero(gt_numpy)
        sample+= non_zero_count
    print("sample:{}".format(sample))


    HR, LR, lr_gt = next(iter(train_loader))
    b, c, h, w = HR.shape
    LR_out = nn.functional.interpolate(HR, scale_factor=(0.25, 0.25), mode='bicubic')  # 4X-->1X
    randn3 = np.random.randint(0, b)
    HR_ = HR[randn3]
    LR_ = LR[randn3]
    lr = LR_out[randn3]
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("LR", fontsize=20)
    plt.imshow(np.transpose(torchvision.utils.make_grid(LR_,
                                                        nrow=2, padding=1, normalize=True).cpu(),
                            (1, 2, 0))[:, :, [60, 31, 12]])

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("HR", fontsize=20)
    plt.imshow(np.transpose(torchvision.utils.make_grid(HR_,
                                                        nrow=2, padding=1, normalize=True).cpu(),
                            (1, 2, 0))[:, :, [60, 31, 12]])
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("lR", fontsize=20)
    plt.imshow(np.transpose(torchvision.utils.make_grid(lr,
                                                        nrow=2, padding=1, normalize=True).cpu(),
                            (1, 2, 0))[:, :, [60, 31, 12]])
    # 添加超级标题
    plt.suptitle("TEST-LR VS HR", fontsize=25)
    plt.show()
    plt.close()


