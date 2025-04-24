import logging

import torch, torchvision
from prettytable import PrettyTable
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from einops import rearrange, repeat
from tqdm.notebook import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math, os, copy

import torch
import torch.nn as nn

from classifier import FeatureExtractor, Classifier
from tools.metrics_utils import *
from tools.my_dataloder import *
from tools.auxil import *
from tools.show_maps import *
from tools.progress_bar import progress_bar

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def calculate_sam(target_data, reference_data):
    # 归一化目标数据和参考数据
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h * w).permute(0, 2, 1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    # 计算点积
    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    # 计算长度乘积
    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    # 计算SAM光谱角
    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.att = att
        self.attn = SelfAtt(dim_out, num_heads=num_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        if self.att:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=32, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = img_size

        # Downsampling stage of U-net
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, noise_level):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


def select_confidence_label(classifier_result, confidence_threshold=0.8):
    classifier_result_probabilities = F.softmax(classifier_result, dim=1)
    # print(classifier_result_probabilities)
    pseudo_pro, pseudo_labels = torch.max(classifier_result_probabilities, dim=1)
    index = torch.where(pseudo_pro > confidence_threshold)[-1]
    if index.shape[0] == 0:
        # print('NO pseudo_labels')
        pseudo_labels = None
    else:
        pseudo = pseudo_pro[index]
        pseudo_labels = pseudo_labels[index]
        # print(index, pseudo, pseudo_labels)
    return index, pseudo_labels


# def sample_points_per_class(gt, mask, sampling_ratio=0.1):
#     classes = torch.unique(gt)  # 获取所有类别
#     classes = classes[classes != 0]  # 去除类别0
#     sample_points = {}
#     for c in classes:
#         points = torch.nonzero((gt == c) & mask, as_tuple=False)  # 获取该类的所有样本点
#         sample_size = int(len(points) * sampling_ratio)  # 计算采样数量
#         sampled_points = points[torch.randperm(len(points))[:sample_size]]  # 随机采样
#         sample_points[c.item()] = sampled_points
#     return sample_points

def data_partition(samples_type, class_count, gt, train_ratio, val_ratio, N=10):
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
        # 为每个类别生成索引
        # for sample in range(gt.shape[0]):  # 遍历每个样本
        #     for cls in range(1, class_count + 1):  # 遍历每个类别
        #         cls_indices = np.transpose(np.nonzero(gt[sample] == cls))
        #         cls_sample_count = min(len(cls_indices), N)  # 确定每个类别的样本数
        #         class_samples_count[cls] = class_samples_count.get(cls, 0) + cls_sample_count
        #         if cls_sample_count > 0:
        #             selected_indices = random.sample(list(cls_indices), cls_sample_count)
        #             train_data_index.update({(sample, idx[0], idx[1]) for idx in selected_indices})
        for cls in range(1, class_count + 1):  # 遍历每个类别
            cls_indices = np.transpose(np.nonzero(gt == cls))  # 获取当前类别的所有索引
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


def target_data_partition(gt):
    # 转换gt为numpy数组（如果它是一个Tensor）
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().detach().numpy()

    # 获取测试样本的标签图
    test_label = np.zeros_like(gt)

    # 找到非背景像元的位置并在测试标签图中设置相应的值
    test_label[gt != 0] = gt[gt != 0]

    return test_label


def gen_cnn_data_hr(data_lrHSI,data_2x, data_pred, patchsize_lrHSI,patchsize_2X, patchsize_pred, test_label):
    data_lrHSI = data_lrHSI.cpu().detach().numpy()
    data_2x = data_2x.cpu().detach().numpy()
    data_pred = data_pred.cpu().detach().numpy()

    num_samples, bands, height_lr, width_lr = data_lrHSI.shape
    _, _, height_2x, width_2x = data_2x.shape
    _, _, height_pred, width_pred = data_pred.shape  # X4预测

    # ##### 给lrHSI和pred_HSI打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding

    # 给 data_lrHSI 和 data_hrHSI 打padding
    pad_width_lr = np.floor(patchsize_lrHSI / 2).astype(int)
    pad_width_2x = np.floor(patchsize_2X / 2).astype(int)
    pad_width_pred = np.floor(patchsize_pred / 2).astype(int)

    data_lrHSI_pad = np.pad(data_lrHSI, ((0, 0), (0, 0), (pad_width_lr, pad_width_lr), (pad_width_lr, pad_width_lr)),
                            mode='symmetric')
    data_2x_pad = np.pad(data_2x, ((0, 0), (0, 0), (pad_width_2x, pad_width_2x), (pad_width_2x, pad_width_2x)),
                         mode='symmetric')

    data_pred_pad = np.pad(data_pred,
                           ((0, 0), (0, 0), (pad_width_pred, pad_width_pred), (pad_width_pred, pad_width_pred)),
                           mode='symmetric')

    # 获取训练标签的坐标
    test_indices = np.where(test_label != 0)
    num_test_samples = len(test_indices[0])

    # 初始化训练集
    TestPatch_lrHSI = np.empty((num_test_samples, bands, patchsize_lrHSI, patchsize_lrHSI), dtype='float32')
    TestPatch_2x = np.empty((num_test_samples, bands, patchsize_2X, patchsize_2X), dtype='float32')
    TestPatch_pred = np.empty((num_test_samples, bands, patchsize_pred, patchsize_pred), dtype='float32')
    TestLabel = np.empty(num_test_samples, dtype=int)

    for i in range(num_test_samples):
        sample, x, y = test_indices[0][i], test_indices[1][i], test_indices[2][i]
        x_lr, y_lr = x // (height_pred // height_lr), y // (width_pred // width_lr)
        x_2x, y_2x = x // (height_pred // height_2x), y // (width_pred // width_2x)

        ind1 = x_lr + pad_width_lr
        ind2 = y_lr + pad_width_lr
        ind3 = x + pad_width_pred
        ind4 = y + pad_width_pred
        ind5 = x_2x + pad_width_2x
        ind6 = y_2x + pad_width_2x


        # 提取低分辨率和高分辨率的patch
        patch_lr = data_lrHSI_pad[sample, :, ind1 - pad_width_lr:ind1 + pad_width_lr + 1,
                   ind2 - pad_width_lr:ind2 + pad_width_lr + 1]

        patch_2x = data_2x_pad[sample, :, ind5 - pad_width_2x:ind5 + pad_width_2x + 1,
                   ind6 - pad_width_2x:ind6 + pad_width_2x + 1]

        patch_pred = data_pred_pad[sample, :, ind3 - pad_width_pred:ind3 + pad_width_pred + 1,
                     ind4 - pad_width_pred:ind4 + pad_width_pred + 1]

        TestPatch_lrHSI[i] = patch_lr
        TestPatch_2x[i] = patch_2x
        TestPatch_pred[i] = patch_pred
        TestLabel[i] = test_label[sample, x, y]

    # 数据转换为PyTorch张量
    TestPatch_lrHSI = torch.from_numpy(TestPatch_lrHSI)
    TestPatch_2x = torch.from_numpy(TestPatch_2x)
    TestPatch_pred = torch.from_numpy(TestPatch_pred)
    TestLabel = torch.from_numpy(TestLabel) - 1
    TestLabel = TestLabel.long()

    # 返回训练集
    return TestPatch_lrHSI,TestPatch_2x, TestPatch_pred, TestLabel


def generate_full_classification_map(classifier_final, test_label):
    """
    Generate the full classification map for the test dataset including background.
    :param classifier_final: The classification result for the non-background pixels.
    :param test_label: The test label including background.
    :return: Full classification map.
    """
    if isinstance(test_label, torch.Tensor):
        test_label = test_label.cpu().detach().numpy()
    # Initialize the full classification map with background label (0)
    full_classification_map = np.zeros_like(test_label, dtype=int)

    # Get indices of non-background pixels in the test dataset
    non_background_indices = np.nonzero(test_label)

    # Convert classifier_final to numpy if it is a torch tensor
    if isinstance(classifier_final, torch.Tensor):
        classifier_final = classifier_final.cpu().detach().numpy()

    # Get the predicted labels for non-background pixels
    predicted_labels = np.argmax(classifier_final, axis=1)

    # Iterate over non-background indices and fill the classification map
    for idx, label in zip(zip(*non_background_indices), predicted_labels):
        full_classification_map[idx] = label + 1  # Assuming class labels start from 1

    return full_classification_map

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class Diffusion(nn.Module):
    def __init__(self, model_1,model_2, device, img_size, LR_size, channels=3, num_classes=9,patch_size=11):
        super().__init__()
        self.channels = channels
        # self.model = model.to(device)
        self.model_1 = model_1.to(device)
        self.model_2 = model_2.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.upsample_2X = nn.Upsample(scale_factor=2, mode='bicubic')
        self.upsample_4X = nn.Upsample(scale_factor=4, mode='bicubic')

        self.classifier_scale=-10     #2可行，3可行
        self.guide_criterion= FocalLoss(gamma=2, weight=None)


    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior

    def p_mean_variance(self, x, t, clip_denoised: bool, condition=None):
        batch_size, h = x.shape[0], x.shape[2]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        # x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))
        if h == 2 * self.LR_size:
            x_start = self.model_1(torch.cat([condition, x], dim=1),
                                   noise_level=noise_level)
        elif h == self.img_size:
            x_start = self.model_2(torch.cat([condition, x], dim=1),
                                 noise_level=noise_level)
        else:
            print("error")
            exit()

        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance, x_start

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img1,img2, t, gt, clip_denoised=True, condition=None):

        # 2x
        mean1, log_variance1, img1_0 = self.p_mean_variance(x=img1, t=t, clip_denoised=clip_denoised,
                                                            condition=self.upsample_2X(condition))

        noise1 = torch.randn_like(img1) if t > 0 else torch.zeros_like(img1)

        # 4X
        mean2, log_variance2, img2_0 = self.p_mean_variance(x=img2, t=t, clip_denoised=clip_denoised,
                                                            condition=self.upsample_2X(img1_0))

        noise2 = torch.randn_like(img2) if t > 0 else torch.zeros_like(img2)


        img1 = mean1 + noise1 * (0.5 * log_variance1).exp()
        img2 = mean2 + noise2 * (0.5 * log_variance2).exp()

        return img1,img2
    @torch.no_grad()
    def p_sample_guide(self, img1,img2, t, gt, clip_denoised=True, condition=None):

        # 2x
        mean1, log_variance1, img1_0 = self.p_mean_variance(x=img1, t=t, clip_denoised=clip_denoised,
                                                            condition=self.upsample_2X(condition))

        noise1 = torch.randn_like(img1) if t > 0 else torch.zeros_like(img1)

        # 4X
        mean2, log_variance2, img2_0 = self.p_mean_variance(x=img2, t=t, clip_denoised=clip_denoised,
                                                            condition=self.upsample_2X(img1_0))

        noise2 = torch.randn_like(img2) if t > 0 else torch.zeros_like(img2)

        # test split
        test_label = target_data_partition(gt)
        # test_label_num = np.count_nonzero(test_label)
        # print('test_label num:',test_label_num,test_label.shape)
        # patch
        if np.count_nonzero(test_label) == 0:  # 检查train_label是否有非零元素
            output_final, TestLabel = None, None
        else:
            TestPatch_lrHSI,TestPatch_2X, TestPatch_pred, TestLabel = gen_cnn_data_hr(condition, img1_0,img2_0, self.patch_size,self.patch_size, self.patch_size, test_label)
            # print('TestPatch_lrHSI-{}, TestPatch_pred-{}, TestLabel-{}'.format(TestPatch_lrHSI.shape, TestPatch_pred.shape, TestLabel.shape))
            # 确保输入数据和模型都在同一个设备上
            TestPatch_lrHSI = TestPatch_lrHSI.to(self.device)
            TestPatch_2X = TestPatch_2X.to(self.device)
            TestPatch_pred = TestPatch_pred.to(self.device)
            TestLabel = TestLabel.to(self.device)  # 每个点对应的 类别 【0----class-1】
            # print(TestLabel.shape)

            # classifier
            output_4x, output_final = classifier_model(TestPatch_lrHSI,TestPatch_2X, TestPatch_pred)

            # 挑选出高置性度伪标签
            index, pseudo_labels = select_confidence_label(output_final, confidence_threshold=0.95)
            if len(index) != 0:
                index = index.to(self.device)
                pseudo_labels = pseudo_labels.to(self.device)

                loss_n = self.guide_criterion(output_4x[index], pseudo_labels)

                mean2 = mean2.float() + self.variance[t] * 2 * loss_n.item() * self.classifier_scale


        img1 = mean1 + noise1 * (0.5 * log_variance1).exp()
        img2 = mean2 + noise2 * (0.5 * log_variance2).exp()

        return img1,img2, output_final, TestLabel

    # Progress whole reverse diffusion process

    @torch.no_grad()
    def sr(self, lrHS, gt):
        img1 = torch.rand_like(self.upsample_2X(lrHS), device=lrHS.device)  # 2X
        img2 = torch.rand_like(self.upsample_4X(lrHS), device=lrHS.device)  # 4X
        for i in reversed(range(0, self.num_timesteps)):

            img1, img2 = self.p_sample(img1,img2, i, gt, condition=lrHS)

        return img1, img2

    @torch.no_grad()
    def sr_and_classification_guide(self, lrHS, gt):
        img1 = torch.rand_like(self.upsample_2X(lrHS), device=lrHS.device)  # 2X
        img2 = torch.rand_like(self.upsample_4X(lrHS), device=lrHS.device)  # 4X
        for i in reversed(range(0, self.num_timesteps)):
            if i>=500:
                img1, img2 = self.p_sample(img1,img2, i, gt, condition=lrHS)
            else:
                img1,img2, classifier_final, TestLabel = self.p_sample_guide(img1,img2, i, gt, condition=lrHS)

        return img2, classifier_final, TestLabel

    # # Compute loss to train the model
    def net(self,hrHS, lrHS, gt):

        hrHS = hrHS
        lrHS = lrHS
        gt = gt

        b, c, h, w = hrHS.shape
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))
        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, hrHS.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(hrHS.device)

        lrHS_up2X = F.interpolate(hrHS, scale_factor=(0.5,0.5), mode='bicubic')
        # 2X
        noise1 = torch.randn_like(self.upsample_2X(lrHS), device=lrHS.device)  # X1---->X2
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy1 = sqrt_alpha * lrHS_up2X + (1 - sqrt_alpha ** 2).sqrt() * noise1
        # 4X
        noise = torch.randn_like(hrHS).to(hrHS.device)  # X1---->X4
        x_noisy = sqrt_alpha * hrHS + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The bilateral model predict actual x0 added at time step t
        # # 2X
        pred_x0_1 = self.model_1(torch.cat([self.upsample_2X(lrHS), x_noisy1], dim=1),
                                 noise_level=sqrt_alpha)
        # 4X
        pred_x0 = self.model_2(torch.cat([self.upsample_2X(pred_x0_1.detach()), x_noisy], dim=1),
                             noise_level=sqrt_alpha)

        loss_1 = self.loss_func(lrHS_up2X, pred_x0_1) / int(b * c * (h/2) * (w/2))
        loss_2 = self.loss_func(hrHS, pred_x0) / int(b * c * h * w)

        return loss_1, loss_2

    def forward(self, hrHS, lrHS, gt,  *args, **kwargs):
        return self.net(hrHS, lrHS, gt, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False, num_classes=9,patch_size=11):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size
        self.num_classes = num_classes
        self.patch_size = patch_size



        # X2
        model_1 = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout,
                     img_size//2)
        # X4
        model_2 = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout,
                     img_size)

        self.sr3 = Diffusion(model_1,model_2, device, img_size, LR_size, out_channel, num_classes=self.num_classes,patch_size=self.patch_size)
        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, insert, epoch, verbose,start_sr):

        train = True
        best_loss_1 = 100  # 2X
        best_loss_2 = 100  # 4X
        for i in range(insert, epoch):
            train_loss = 0
            self.sr3.train()
            loss_1_epoch = 0
            loss_2_epoch = 0
            if train:  # 2x-4X
                print('epoch: {}'.format(i + 1))
                logging.info('epoch: {}'.format(i + 1))
                for step, [hrHS, lrHS, gt] in enumerate(self.dataloader):
                    #
                    hrHS = hrHS.to(self.device).type(torch.float32)
                    lrHS = lrHS.to(self.device).type(torch.float32)
                    gt = gt.to(self.device).type(torch.long)
                    b, c, h, w = hrHS.shape
                    self.optimizer.zero_grad()
                    loss_1, loss_2  = self.sr3(hrHS, lrHS, gt)

                    loss_1 = loss_1.sum()
                    loss_2 = loss_2.sum()

                    if i<start_sr:
                        loss = loss_1
                    else:
                        loss = loss_1 + loss_2
                    loss.backward()

                    self.optimizer.step()

                    loss_1_epoch = loss_1.item() + loss_1_epoch
                    loss_2_epoch = loss_2.item() + loss_2_epoch

                    train_loss += loss.item()
                    progress_bar(step, len(self.dataloader), 'Loss: %.4f' % (train_loss / (step + 1)))


                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                x.add_column("loss_1", [loss_1_epoch / float(len(self.dataloader))])
                x.add_column("loss_2", [loss_2_epoch / float(len(self.dataloader))])
                print(x)
                logging.info('损失函数:\n%s', x)
            if (i + 1) % 10 == 0:
                if i<start_sr:
                    if train_loss <= best_loss_1:
                        best_loss_1 = train_loss
                        self.save(self.save_path, 1)  # 2X
                else:
                    if train_loss <= best_loss_2:
                        best_loss_2 = train_loss
                        self.save(self.save_path, 2)  #2X+4X

            if (i + 1) % verbose == 0:
                print('begin test')
                print('{:-^20}'.format(i + 1))
                logging.info('begin test')
                logging.info('{:-^20}'.format(i + 1))

                self.sr3.eval()
                # test part one
                test_data = copy.deepcopy(next(iter(self.testloader)))
                [hrHS, lrHS, gt] = test_data
                hrHS = hrHS.to(self.device).type(torch.float32)
                lrHS = lrHS.to(self.device).type(torch.float32)
                gt = gt.to(self.device)

                b, c, h, w = hrHS.shape
                randn3 = np.random.randint(0, b)
                # test loss
                with torch.no_grad():
                    val_loss_1, val_loss_2 = self.sr3(hrHS, lrHS, gt)
                    val_loss_1 = val_loss_1.sum()
                    val_loss_2 = val_loss_2.sum()
                    if i<start_sr:
                        val_loss = val_loss_1
                    else:
                        val_loss = val_loss_1 + val_loss_2
                # test part
                img1, result_sr = self.test(hrHS, lrHS, gt)

                PSNR = psnr(result_sr.cpu().detach().numpy(), hrHS.cpu().detach().numpy())
                hrHS_ = hrHS[randn3]
                lrHS_ = lrHS[randn3]
                result_sr_ = result_sr[randn3]

                # Transform to low-resolution images
                # Save example of test images to check training
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.title("LR-HSI", fontsize=20)
                plt.imshow(np.transpose(torchvision.utils.make_grid(lrHS_,
                                                                    nrow=2, padding=1, normalize=True).cpu(),
                                        (1, 2, 0))[:, :, [22, 10, 3]])

                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.title("HR-HSI", fontsize=20)
                plt.imshow(np.transpose(torchvision.utils.make_grid(hrHS_,
                                                                    nrow=2, padding=1, normalize=True).cpu(),
                                        (1, 2, 0))[:, :, [22, 10, 3]])
                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.title("SR-HSI", fontsize=20)
                plt.imshow(np.transpose(torchvision.utils.make_grid(result_sr_,
                                                                    nrow=2, padding=1, normalize=True).cpu(),
                                        (1, 2, 0))[:, :, [22, 10, 3]])


                plt.suptitle("epoch-{}_PSNR-{:.4f}".format(i + 1, PSNR), fontsize=25)
                plt.show()
                plt.close()

                print('评价指标:')
                y = PrettyTable()
                y.add_column("Index", ['value'])
                y.add_column('val_loss', [val_loss])
                y.add_column('val_loss_1', [val_loss_1.item()])
                y.add_column('val_loss_2', [val_loss_2.item()])
                y.add_column("PSNR", [PSNR])
                print(y)
                logging.info('评价指标:\n%s', y)

    def test(self, hrHS, lrHS, gt):
        hrHS = hrHS
        lrHS = lrHS
        gt = gt
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                img1,result_SR = self.sr3.module.sr(lrHS, gt)
            else:
                img1,result_SR = self.sr3.sr(lrHS, gt)
        # self.sr3.train()
        return img1,result_SR

    def test_guide(self, hrHS, lrHS, gt):
        hrHS = hrHS
        lrHS = lrHS
        gt = gt
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR,classifier_final, TestLabel = self.sr3.module.sr_and_classification_guide(lrHS, gt)
            else:
                result_SR,classifier_final, TestLabel = self.sr3.sr_and_classification_guide(lrHS, gt)
        # self.sr3.train()
        return result_SR,classifier_final, TestLabel

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + 'SR3_model_epoch-{}.pth'.format(i))

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")



class myclassifier(nn.Module):
    def __init__(self,channels=128, num_classes = 16):
        super(myclassifier, self).__init__()
        self.FeatureExtractor_LR = FeatureExtractor(in_channels=channels)
        self.FeatureExtractor_2X = FeatureExtractor(in_channels=channels)
        self.FeatureExtractor_4X = FeatureExtractor(in_channels=channels)

        self.Classifier_4x = Classifier(input_features=128, num_classes=num_classes)
        self.Classifier_final = Classifier(input_features=128*3, num_classes=num_classes)



    def forward(self, data_lrHSI,data_2x,  data_pred):
        # 提取特征
        feature_lr = self.FeatureExtractor_LR(data_lrHSI)
        feature_2x = self.FeatureExtractor_2X(data_2x)
        feature_pred = self.FeatureExtractor_4X(data_pred)

        # 分类
        output_4x = self.Classifier_4x(feature_pred)
        cat_feat = torch.cat([feature_lr, feature_2x, feature_pred], dim=1)
        output_final = self.Classifier_final(cat_feat)

        return output_4x, output_final

if __name__ == "__main__":
    data_path = './data'
    save_path = './model/model_PC_SR/'
    save_path_classifier = './model/model_PC_classifier/'
    load_path = './model/model_PC_SR/SR3_model_epoch-2.pth'
    save_path_predHR = './result/guide_pc/predHR/'
    save_path_gtHR = './result/guide_pc/gtHR/'
    save_path_predC = './result/guide_pc/save_path_predC/'
    save_path_predmap = './result/guide_pc/predmap/'
    save_path_classification_data = './data_classification/pc/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_predHR):
        os.makedirs(save_path_predHR)
    if not os.path.exists(save_path_gtHR):
        os.makedirs(save_path_gtHR)
    if not os.path.exists(save_path_predC):
        os.makedirs(save_path_predC)
    if not os.path.exists(save_path_predmap):
        os.makedirs(save_path_predmap)
    if not os.path.exists(save_path_classification_data):
        os.makedirs(save_path_classification_data)
    if not os.path.exists(save_path_classifier):
        os.makedirs(save_path_classifier)

    name = 'PC'
    lr_spatial_size = 32
    up_scale = 4
    train_percent_sr = 0.6
    train_num_classifier = 100
    patch_size = 11
    batch_size_sr = 1
    batch_size_classifier = 256
    lr = 1e-5
    seed = 123
    epochs_classifier = 300
    num_experiments = 10
    filename = 'PC-guide'
    logging.basicConfig(filename=filename, level=logging.INFO)

    set_seed(seed)

    train_loader, test_loader, num_classes, bands = load_hyper_sr(data_path, name, lr_spatial_size=lr_spatial_size,
                                                                     up_scale=up_scale,
                                                                     train_percent=train_percent_sr,train_num = train_num_classifier,
                                                                     batch_size=batch_size_sr, components=None,
                                                                     rand_state=seed, filename=filename)
    print("----------数据加载完毕--------")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if cuda else "cpu")
    schedule_opt = {'schedule': 'cosine', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    # 加载分类模型
    classifier_model = myclassifier(channels=bands, num_classes=num_classes)
    model_classifier = torch.load(save_path_classifier+'best_model_epoch.pth')
    classifier_model.load_state_dict(model_classifier)
    classifier_model.to(device)  # 将模型移动到正确的设备
    classifier_model.eval()  # 设置模型为评估模式

    sr3 = SR3(device, img_size=lr_spatial_size * up_scale, LR_size=lr_spatial_size, loss_type='l1',
              dataloader=train_loader, testloader=test_loader,
              schedule_opt=schedule_opt,
              save_path=save_path,
              load_path=load_path, load=True,
              in_channel=bands * 2, out_channel=bands,
              inner_channel=256,
              norm_groups=16, channel_mults=(1, 2, 4, 4), dropout=0.2, res_blocks=2, lr=1e-4, distributed=False,
              num_classes=num_classes,patch_size=patch_size)
    # sr3.train(insert=6000, epoch=21000, verbose=100, start_sr=4000)
    # sr3.train(insert=1000, epoch=3000, verbose=50)


    print('collecting test results...')
    print("get dataset for classification...")
    total_time = 0
    PSNRs = np.ones((len(test_loader))) * -1000.0
    accs=np.ones((len(test_loader))) * -1000.0
    predicted = []   #预测标签
    Gt = []          #真实标签
    pre_map_all = []  #用于制图
    for idx, (hrHS, lrHS, gt) in enumerate(test_loader):
        with torch.no_grad():
            print('{:-^20}'.format(idx + 1, len(test_loader)))
            start = time.time()
            hrHS = hrHS.to(device).type(torch.float32)
            lrHS = lrHS.to(device).type(torch.float32)
            gt = gt.to(device).type(torch.long)
            b, c, h, w = hrHS.shape
            randn3 = np.random.randint(0, b)
            result_sr,classifier_final, TestLabel = sr3.test_guide(hrHS, lrHS, gt)
            predHR = result_sr.cpu().detach().numpy()
            gtHR = hrHS.cpu().detach().numpy()

            sio.savemat(
                os.path.join(save_path_predHR, '%d.mat' % (idx + 1)),
                {'predHR': predHR})
            sio.savemat(
                os.path.join(save_path_gtHR, '%d.mat' % (idx + 1)),
                {'gtHR': gtHR})

            end = time.time()
            time_use = end - start
            total_time += time_use
            print('test complete! total time :{}'.format(time_use))

            # classfication
            if classifier_final is not None and TestLabel is not None:
                print("result_sr-{},  classifier_final-{},TestLabel-{}".format(result_sr.shape, classifier_final.shape,
                                                                          TestLabel.shape))
                predicted.append(classifier_final.data.cpu().detach().numpy())
                Gt.append(TestLabel.data.cpu().detach().numpy())

                classification_map = generate_full_classification_map(classifier_final, gt)
                pre_map_all.append(classification_map)
                pre_map = classification_map

                sio.savemat(os.path.join(save_path_predC, '%d.mat' % (idx + 1)),{'pre_map': pre_map})

                print("classification_map-{}".format(classification_map.shape))
                logging.info("classification_map-{}".format(classification_map.shape))

                acc = accuracy(classifier_final.data, TestLabel.data)[0].item()
                gt_ = gt[randn3]
                pred_ = classification_map[randn3]
            else:
                classification_map = gt.cpu().detach().numpy()
                pre_map_all.append(classification_map)
                pre_map = classification_map
                sio.savemat(os.path.join(save_path_predC, '%d.mat' % (idx + 1)),{'pre_map': pre_map})
                acc = 100

            # 计算评价指标
            PSNR = psnr(predHR, gtHR)
            PSNRs[idx] = PSNR
            accs[idx] = acc

            # 显示部分图

            hrHS_ = hrHS[randn3]
            lrHS_ = lrHS[randn3]
            result_sr_ = result_sr[randn3]

            # Transform to low-resolution images
            # Save example of test images to check training
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 3, 1)
            plt.axis("off")
            plt.title("LR-HSI", fontsize=20)
            plt.imshow(np.transpose(torchvision.utils.make_grid(lrHS_,
                                                                nrow=2, padding=1, normalize=True).cpu(),
                                    (1, 2, 0))[:, :, [60, 31, 12]])

            plt.subplot(1, 3, 2)
            plt.axis("off")
            plt.title("HR-HSI", fontsize=20)
            plt.imshow(np.transpose(torchvision.utils.make_grid(hrHS_,
                                                                nrow=2, padding=1, normalize=True).cpu(),
                                    (1, 2, 0))[:, :, [60, 31, 12]])
            plt.subplot(1, 3, 3)
            plt.axis("off")
            plt.title("4X-HSI", fontsize=20)
            plt.imshow(np.transpose(torchvision.utils.make_grid(result_sr_,
                                                                nrow=2, padding=1, normalize=True).cpu(),
                                    (1, 2, 0))[:, :, [60, 31, 12]])

            plt.suptitle("idx-{}_PSNR-{:.4f}".format(idx + 1, PSNR), fontsize=25)
            plt.show()
            plt.close()

            print('评价指标:')
            y = PrettyTable()
            y.add_column("Index", ['value'])
            y.add_column("PSNR", [PSNR])
            print(y)

    print('all psnr:', np.average(PSNRs))


    predicted_all_indices = []
    Gt_all_indices = []
    for i in range(len(predicted)):
        predicted_indices= np.argmax(predicted[i], axis=1)
        predicted_all_indices.append(predicted_indices)
        Gt_all_indices.append(Gt[i])
    predicted_all_indices = np.concatenate(predicted_all_indices)
    Gt_all_indices = np.concatenate(Gt_all_indices)

    classification, confusion, result = reports(
        predicted_all_indices,
        Gt_all_indices,
        'PU'
    )
    # 将结果添加到相应的列表中
    print(classification)
    print(confusion)
    print(
        "\nOA:{},AA:{},Kappa:{}\neach_class_acc:{}".format(result[0], result[1], result[2],
                                                           result[3:]))
    target_gt = np.zeros((1096, 715)).astype('long')
    # print('target_gt',target_gt.shape)
    recon_gt = get_all_gt(target_gt, [lr_spatial_size*up_scale,lr_spatial_size*up_scale], pre_map_all)
    print(type(recon_gt), recon_gt.shape)
    show_label(target_gt, recon_gt, num_classes,
               os.path.join(save_path_predmap, name + '_gt_recon_128.png'))