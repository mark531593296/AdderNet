# Pytorch 0.4.0 VGG16实现cifar10分类.
# @Time: 2018/6/23
# @Author: wxq

import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import adder
#model_path = './model_pth/vgg16_bn-6c64b313.pth'  # 预训练模型的数据储存文件
# 当时LR=0.01遇到问题，loss逐渐下降，但是accuracy并没有提高，而是一直在10%左右，修改LR=0.00005后，该情况明显有所好转，准确率最终提高到了
# 当LR=0.0005时，发现准确率会慢慢提高，但是提高的速度很慢，这时需要增加BATCH_SIZE，可以加快训练的速度，但是要注意，BATCH_SIZE增大会影响最终训练的准确率，太大了还可能也会出现不收敛的问题
# 另外，注意每次进入下一个EPOCH都会让准确率有较大的提高，所以EPOCH数也非常重要，需要让网络对原有数据进行反复学习，强化记忆
#
# 目前，调试的最好的参数是BATCH_SIZE = 500  LR = 0.0005  EPOCH = 10  最终准确率为：69.8%    用时：
# BATCH_SIZE = 500  # 将训练集划分为多个小批量训练，每个小批量的数据量为BATCH_SIZE
# LR = 0.0005  # learning rate
# EPOCH = 10  # 训练集反复训练的次数，每完整训练一次训练集，称为一个EPOCH
# CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def conv3x3(in_channels, out_channels, padding=1):
    """3x3 convolution with padding"""
    return adder.adder2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):  # 构造函数 num_class根据最后分类的种类数量而定，cifar为10所以这里是10
        super(VGG, self).__init__()  # pytorch继承nn.Module模块的标准格式，需要继承nn.Module的__init__初始化函数
        self.features = features  # 图像特征提取网络结构（仅包含卷积层和池化层，不包含分类器）
        self.classifier = nn.Sequential(  # 图像特征分类器网络结构
            # FC4096 全连接层
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # FC4096 全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # FC1000 全连接层
            nn.Linear(4096, num_classes))
        # 初始化各层的权值参数
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']  # vgg16的网络结构参数，数字代表该层的卷积核个数，'M'代表该层为最大池化层


def make_layers(cfg, batch_norm=False):
    """利用cfg，生成vgg网络每层结构的函数"""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':  # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 根据cfg设定卷积层的卷积核的数量v，根据论文，vgg网络中的卷积核尺寸均使用3x3xn，n是输入数据的通道数
            conv2d = conv3x3(in_channels, v, padding=1)  # 卷积层,in_channels是输入数据的通道数，初始RGB图像的通道数为3
            if batch_norm:  # 对batch是否进行标准化
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]  # 每次卷积完成后，需要使用激活函数ReLU激活一下，保持特征的非线性属性
            in_channels = v  # 下一层的输入数据的通道数，就是上一层卷积核的个数
    return nn.Sequential(*layers)  # 返回一个包含了网络结构的时序容器，加*是为了只传递layers列表中的内容，而不是传递列表本身


def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)  # batch_norm一定要等于True，如果不对batch进行标准化，那么训练结果的准确率一直无法提升
    # model.load_state_dict(torch.load(model_path))  # 如果需要使用预训练模型，则加入该代码
    return model





