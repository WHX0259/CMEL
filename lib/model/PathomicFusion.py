

import math
import torch.nn as nn
# Numerical / Array
import numpy as np
import torch
from torch.nn import init, Parameter
from .fusion import *
from torch.nn import init, Parameter
from .TextNet import TextNetFeature
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import init
import time
from einops import rearrange
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.flatten = nn.Flatten()  # 扁平化层
        self.fc = nn.Linear(dim, num_classes)  # 全连接层
 
    def forward(self, x):
        x = self.global_avg_pool(x)  # 应用全局平均池化
        x = self.flatten(x)  # 扁平化特征图
        x = self.fc(x)  # 应用全连接层得到类别预测
        return x

class Attention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.relu(x1 + x2 + x3)
        return y
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)
#我要不就先用ResNeXt结构进行替换
#我希望我的encoder或者Backbone是输出各层特征的
class ResNeXt_encoder(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4):
        super(ResNeXt_encoder, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        return feat_list#获得特征列表
# from options import parse_args#这个是参数文件
#自己改进吧，全都换成自己的
# from utils import *
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


def define_trifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=3, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion_A':
        fusion = TrilinearFusion_A(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'pofusion_B':
        fusion = TrilinearFusion_B(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion

class PathomicNet(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, base_width=64, widen_factor=4, text_dim=25, feat_dim=128):#这个应该是参数，act是不是激活函数，k是什么
        super(PathomicNet, self).__init__()
        self.clinic_net = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.img_net = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.stages = self.img_net.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fusion = define_bifusion(fusion_type='pofusion', skip=1, use_bilinear=1,
                                       gate1=1, gate2=1, dim1=feat_dim, dim2=64, #dim1,dim2是输入的特征
                                       scale_dim1=2, scale_dim2 = 1, mmhid=feat_dim, 
                                       dropout_rate=0.2)
        self.classifier =  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
    def forward(self, img, clinic):
        img_feature_list = self.img_net(img)
        if clinic.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            clinic = clinic.unsqueeze(1)
        clin_vec = self.clinic_net(clinic)
        clin_vec = clin_vec.view(clin_vec.size(0), -1)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 

        img_vec = self.proj(feat_last)#8,128
        features = self.fusion(img_vec, clin_vec)
        output = self.classifier(features)

        return {
            'logits': output.unsqueeze(1),
            'output': output,
            'features': features,
        }
import os
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights

class PathgraphomicNet(nn.Module):#没有使用图卷积主要是肿瘤区域与细胞不同没有点，使用图不太行我觉得
    def __init__(self, cardinality=8, depth=29, nlabels=2, base_width=64, widen_factor=4, text_dim=25, radiomic_dim = 22,feat_dim=128):#这个应该是参数，act是不是激活函数，k是什么
        super(PathgraphomicNet, self).__init__()
        self.clinic_net = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.radiomics_net = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=radiomic_dim)
        self.img_net = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.stages = self.img_net.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fusion = define_trifusion(fusion_type='pofusion_A', skip=1, use_bilinear=1, gate1=1, 
                                       gate2=1, gate3=3, dim1=feat_dim, dim2=64, dim3=64, scale_dim1=2, 
                                       scale_dim2=1, scale_dim3=1, mmhid=feat_dim, dropout_rate=0.25)
        self.classifier =  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
    def forward(self, img, clinic, radiomics):
        img_feature_list = self.img_net(img)
        if clinic.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            clinic = clinic.unsqueeze(1)
        clin_vec = self.clinic_net(clinic)
        clin_vec = clin_vec.view(clin_vec.size(0), -1)
        if radiomics.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            radiomics = radiomics.unsqueeze(1)
        radiomics_vec = self.radiomics_net(radiomics)
        radiomics_vec = radiomics_vec.view(radiomics_vec.size(0), -1)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 

        img_vec = self.proj(feat_last)#8,128
        features = self.fusion(img_vec, clin_vec,radiomics_vec)
        output = self.classifier(features)

        return {
            'logits': output.unsqueeze(1),
            'output': output,
            'features': features,
        }