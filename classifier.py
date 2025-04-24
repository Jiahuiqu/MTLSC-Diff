# -*- coding: UTF-8 -*-
'''
@Project ：MTLSC-Diff - all 
@File    ：classifier.py
@Author  ：xiaoliusheng
@Date    ：2023/11/15/015 22:07 
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
classifier and feature_extract
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights

# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = ResBlock(128, 64, stride=1)
        self.layer2 = ResBlock(64, 64, stride=1)
        self.layer3 = ResBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return out

class Classifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out






class PretrainedResnetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(PretrainedResnetFeatureExtractor, self).__init__()
        # 加载预训练的ResNet18模型
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 修改第一个卷积层的输入通道数
        resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return out

class PretrainedResnet50FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(PretrainedResnet50FeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 修改第一个卷积层的输入通道数
        resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return out






if __name__ == "__main__":
    # 实例化特征提取器和分类器
    num_channels = 103  # 高光谱数据的波段数
    num_classes = 10  # 假设有 10 个类别
    # feature_extractor = FeatureExtractor(num_channels)
    feature_extractor = PretrainedResnetFeatureExtractor(in_channels=num_channels)
    # classifier = Classifier(input_features=128, num_classes=num_classes)
    classifier = Classifier(input_features=512, num_classes=num_classes)
    x = torch.randn(10,103,11,5)
    fea = feature_extractor(x)
    print(fea.shape)
    out = classifier(fea)
    print(out.shape)



    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001)
