
import os, sys
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
import argparse
import time
import warnings
from einops import rearrange
from .TextNet import TextNetFeature
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

class CrossModalAttention(nn.Module):#对齐再融合
    def __init__(self, img_dim, clinical_dim, output_dim):
        super(CrossModalAttention, self).__init__()
        self.clinical_query = nn.Linear(clinical_dim, output_dim)  # 图像特征的 Query
        self.img_key = nn.Linear(img_dim, output_dim)  # 临床特征的 Key
        self.img_value = nn.Linear(img_dim, output_dim)  # 临床特征的 Value
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img_features, clinical_features):
        # 图像特征 Query
        query = self.clinical_query(clinical_features)
        
        # 临床特征 Key 和 Value
        key = self.img_key(img_features)#8，128
        value = self.img_value(img_features)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = self.softmax(attention_scores)
        
        # 使用注意力权重对临床特征进行加权
        aligned_clinical_features = torch.matmul(attention_weights, value)
        
        # 将对齐后的特征与图像特征进行加和
        output = query + aligned_clinical_features
        return output#进行对齐了
def generate_attention_sequence(num_experts, num_features):
    sequence = []
    sequence.append(num_features-1)
    if num_experts > 1:
        for i in range(num_experts-1):
            sequence.append((num_features - 2 - i % (num_features - 1)) % num_features)
    return sequence

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, stride=None):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")
        if stride==None:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=int(F_l/F_g), padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )
        else:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x1 * psi+x1#这个只是attention得到的加入一个残差链接把


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

# # 生成伪造的测试数据
def generate_dummy_data(batch_size, img_channels, img_height, img_width, clinical_data_size):
    # 生成图像数据
    image_data = torch.randn(batch_size, img_channels, img_height, img_width)  # 随机生成图像数据

    # 生成临床数据
    clinical_data = torch.randn(batch_size, clinical_data_size)  # 随机生成临床数据

    return image_data, clinical_data
#看看把模型变成可以拆解的，不知道要写的代码多不多，要不就分别写出不同的模型代码，
#所有没有使用MoE的的main和其他的就不一样
class ClinicalImageBaseClusterDistancePlusGatingModelA(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelA, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU())

        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        image_feat, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(image_feat, text_feat)
        outputs = self.fusion_fc(fuse_features)
    
        return {
            'logits': outputs.unsqueeze(1),
            'output': outputs,
            'features': fuse_features,
        }
class ClinicalImageBaseClusterDistancePlusGatingModelB(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelB, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.attention = Attention_block(self.stages[-1], self.stages[-1], 128)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.stages[-1], 128), nn.ReLU(), 
        #     nn.Linear(128, 128), nn.ReLU()
        # )
        self.fuse_proj = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU())

        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        # features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        image_features = self.attention(image_features, image_features)
        image_features = F.avg_pool2d(image_features, image_features.size()[3]).view(image_features.size(0), -1) 
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(features[-1], text_feat)
        outputs = self.fusion_fc(fuse_features)
    
        return {
            'logits': outputs.unsqueeze(1),
            'output': outputs,
            'features': fuse_features,
        }

class ClinicalImageBaseClusterDistancePlusGatingModelC(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelC, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.attention = Attention_block(self.stages[-1], self.stages[-1], 128)
        self.cross_attention = CrossModalAttention(128, 64, feat_dim)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.stages[-1], 128), nn.ReLU(), 
        #     nn.Linear(128, 128), nn.ReLU()
        # )
        # self.fuse_proj = nn.Sequential(
        #     nn.Linear(192, 128), nn.ReLU(), 
        #     nn.Linear(128, 128), nn.ReLU())

        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        # features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        image_features = self.attention(image_features, image_features)
        image_features = F.avg_pool2d(image_features, image_features.size()[3]).view(image_features.size(0), -1) 
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        
        aligned_features = self.cross_attention(image_features, text_features)
        # aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(features[-1], text_feat)
        outputs = self.fusion_fc(fuse_features)
    
        return {
            'logits': outputs.unsqueeze(1),
            'output': outputs,
            'features': fuse_features,
        }

class ClinicalImageBaseClusterDistancePlusGatingModelD(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelD, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        # features_flatten = self.proj(feat_last)
        # features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, features, text_features):
        #DAF模块
        index = generate_attention_sequence(self.num_experts, len(features))
        attentioned_features = []
        feat_last = features[-1]
        for idx in range(len(index)):
            att_feat = self.attention[idx](features[index[idx]], feat_last)
            attentioned_features.append(att_feat)
        img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        aligned_features = []
        text_features = text_features.view(text_features.size(0), -1)
        for idx in range(self.num_experts):
            aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
            aligned_features.append(aligned_feat)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(features, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        output = outputs_tensor.mean(dim=1)
    
        return {
            'logits': outputs_tensor,
            'output': output,
            # 'features': fuse_features,
        }

class ClinicalImageBaseClusterDistancePlusGatingModelE(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelE, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.init_itr = 0
        self.k = k
        # 动量模型
        self.encoder_m = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.proj_m = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )

        self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        # 专家输出层
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        self.cluster_init_type = cluster_init_type
        # self.gating = GatingNetwork(feat_dim,num_experts=num_experts)
        #多少次更新聚类中心
        # 定义损失函数，例如交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # 添加一个属性来跟踪迭代次数
        self.iteration_counter = 0
        self.num_iterations = num_iterations
        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.proj,self.proj_m],
                           ]
        self.copy_params()
        # 定义每多少次迭代更新一次聚类中心
        self.update_frequency = 100
        # 队列
        self.queue_size = queue_size
        self.n_clusters = num_experts
        
        self.register_buffer("feat_queues", torch.zeros(num_experts, feat_dim, queue_size))
        self.feat_queues = nn.functional.normalize(self.feat_queues, dim=1)
        self.register_buffer("queue_ptrs", torch.zeros(num_experts, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim))
        # self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim*num_experts))
        self.if_init = False
    def merge_feat_queues(self):
        """
        Merge all feature queues into a single tensor.
        """
        # merged_features = []
        feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
        feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
        feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
        queue_count, queue_index = self.select_effective_features()
        feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
        feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
        feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
        
        # 拼接所有特征，并沿第一维（样本数）拼接
        merged_features = torch.cat((feat1_effect_queue, feat2_effect_queue, feat3_effect_queue), dim=1)

        # 转置拼接后的特征矩阵
        return merged_features.t()
    def select_effective_features(self):
        non_zero_indices = []
        non_zero_counts = []
        for i in range(self.num_experts):
            # 检查feat_queues[i,k,:]是否全为0，使用torch.any结合非等于0的条件
            # 检查每一列是否全部为零
            non_zero = torch.any(self.feat_queues[i] != 0, dim=0)  # shape: (96,)

            # 获取有效的特征索引
            non_zero_indices_i = torch.nonzero(non_zero, as_tuple=True)[0]  # shape: (N,)
            non_zero_indices.append(non_zero_indices_i)
            non_zero_counts.append(len(non_zero_indices_i))
        return non_zero_counts, non_zero_indices
        # print("Non-zero feature indices:", non_zero_indices)
        # print("Count of non-zero features:", non_zero_counts)
    
    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    @torch.no_grad()
    def extract_features_m(self, image):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder_m(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj_m(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        return features_flatten, img_feature_list, 
    def fuse_features(self, features, text_features):
        #DAF模块
        index = generate_attention_sequence(self.num_experts, len(features))
        attentioned_features = []
        feat_last = features[-1]
        for idx in range(len(index)):
            att_feat = self.attention[idx](features[index[idx]], feat_last)
            attentioned_features.append(att_feat)
        img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        aligned_features = []
        text_features = text_features.view(text_features.size(0), -1)
        for idx in range(self.num_experts):
            aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
            aligned_features.append(aligned_feat)
        return aligned_features
    # def momentum_forward(self, image, clinical_data):
    #     return self.extract_and_align_features(image, clinical_data, self.encoder_m)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def update_queues(self, features, pseudo_labels):
        for i in range(self.num_experts):
            ptr = int(self.queue_ptrs[i].item())
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
            idx = (pseudo_labels_tensor == i).nonzero(as_tuple=True)[0]
            # feat_i = torch.gather(features, 1, pseudo_labels_tensor)#16,384
            # feat_i = features[pseudo_labels == i]#这个就是更新
            if len(idx) > 0:
                # 根据索引获取特征
                feat_i = features[idx]
            # if feat_i.size(0) > 0:
                # Update the queue with new features
                num_features = min(feat_i.size(0), self.queue_size - ptr)
                self.feat_queues[i, :, ptr:ptr + num_features] = feat_i[:num_features].t()
                # Move the pointer
                ptr = (ptr + num_features) % self.queue_size
                self.queue_ptrs[i] = ptr

    @torch.no_grad()
    def reassign_queue_features(self):
        # 遍历每个专家队列
        for i in range(self.num_experts):
            # 获取当前队列中的有效特征（非空部分）
            valid_idx = self.queue_ptrs[i].item() if self.queue_ptrs[i].item() > 0 else self.queue_size
            queue_features = self.feat_queues[i, :, :valid_idx].clone().detach().t()  # shape: (valid_size, feat_dim*num_experts)
            
            # 计算每个特征到所有聚类中心的距离
            distances = torch.cdist(queue_features, self.cluster_centers)
            
            # 找到每个特征最近的聚类中心
            new_pseudo_labels = torch.argmin(distances, dim=1)
            
            # 根据新的伪标签重新分配特征到新的队列
            for j in range(self.num_experts):
                # 找到需要分配到队列 j 的特征
                features_to_assign = queue_features[new_pseudo_labels == j]
                
                # 更新队列 j 中的特征
                ptr = int(self.queue_ptrs[j].item())
                num_features = min(features_to_assign.size(0), self.queue_size - ptr)
                
                if num_features > 0:
                    # 更新队列 j 中的特征
                    self.feat_queues[j, :, ptr:ptr + num_features] = features_to_assign[:num_features].t()
                    
                    # 更新指针位置，指向第一个空位置
                    self.queue_ptrs[j] = (ptr + num_features) % self.queue_size
                
                # 如果本次重新分配的特征数量不足以填满队列，则将指针设置为有效的最大位置
                if num_features < features_to_assign.size(0):
                    self.queue_ptrs[j] = self.queue_size

    def entropy_regularization(self, labels):
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        probabilities = cluster_counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    @torch.no_grad()
    def update_cluster_centers(self):
        # Update the cluster centers by averaging the features in each queue
        for i in range(self.num_experts):
            self.cluster_centers[i] = self.feat_queues[i].mean(dim=1).clone().detach()
            self.cluster_centers[i] = nn.functional.normalize(self.cluster_centers[i], dim=0)
    def initialize_cluster_centers(self):
        if self.cluster_init_type == "kmeans":
            self.update_cluster_centers()
            self.kmeans_iteration()
        elif self.cluster_init_type == "kmeans++":
            self.kmeans_plus_plus_init()
            self.kmeans_iteration()
    #使用feature向量的密度进行加权，首先要计算密度,防止聚类坍塌
    def calculateDensityWeight(self, features):
        #计算密度
        merge_feature = self.merge_feat_queues()
        distances = torch.cdist(features, merge_feature)

        # 找到每个输入特征最近的k个邻居的距离
        _, topk_indices = torch.topk(distances, k=self.k, largest=False, sorted=True)
        topk_distances = torch.gather(distances, 1, topk_indices)

        # 计算密度，这里简单地取距离的倒数作为密度的度量
        # 可以考虑对距离进行规范化或者使用其他方法来计算密度
        densities = 1.0 / (topk_distances.sum(dim=1) + 1e-8)  # 加上一个小数值防止除以零
        weights = F.softmax(-densities, dim=0)
        return weights
    def kmeans_plus_plus_init(self):
        """
        使用 KMeans++ 算法初始化聚类中心
        """
        # 合并特征队列
        merge_feature = self.merge_feat_queues()
        
        # 随机选择第一个聚类中心
        centers = [merge_feature[torch.randint(0, merge_feature.size(0), (1,)).item()]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到已选择聚类中心的最近距离
            # 使用torch.min和广播机制避免for循环，提高效率
            distances = torch.stack([torch.norm(merge_feature - c, dim=1).pow(2) for c in centers], dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离计算选择下一个中心的概率
            probabilities = min_distances / min_distances.sum()
            
            # 生成累积概率分布
            cumulative_probabilities = probabilities.cumsum(dim=0)
            
            # 按照概率选择下一个聚类中心
            r = torch.rand(1).item()
            i = torch.where(cumulative_probabilities >= r)[0][0].item()
            centers.append(merge_feature[i])

        # 将中心列表转换为torch张量并赋值给cluster_centers
        self.cluster_centers = torch.nn.Parameter(torch.stack(centers))

    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")

    def generate_pseudo_labels(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
        distances = torch.cdist(features, self.cluster_centers)
        pseudo_labels = torch.argmin(distances, dim=1)  # 选择最近的聚类中心作为标签
        return pseudo_labels
    def calculate_diversity_loss(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
         # 获取当前特征所在的设备
        device = features.device

        # 如果聚类中心在CPU上，将其移动到与features相同的设备
        cluster_centers_on_device = self.cluster_centers.to(device)

        # 计算样本到各个聚类中心的距离
        distances = torch.cdist(features, cluster_centers_on_device)

        # 归一化每个样本的距离
        normalized_distances = distances / distances.sum(dim=1, keepdim=True)

        # 计算方差
        variance = torch.var(normalized_distances, dim=1).mean()

        # 返回多样性损失（负方差）
        return -variance
    def cluster_distance_loss(self):
        # 防止聚类坍塌，最小化聚类中心之间的距离
        dist_loss = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                dist_loss += 1 / (dist + 1e-6)  # 防止除零
        return dist_loss
    def forward(self, image, text):
        # 提取特征并对齐
        features_flatten, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]
        # features = torch.stack(features, dim=1)#B,N,feature_dim

        # features_m = torch.stack(features_m, dim=1)
        if self.if_init == False:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
            bs = features_m_flatten.size(0)
            pseudo_labels_m = random.choices([0, 1, 2], k=bs)
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.init_itr +=1
            if self.init_itr > self.queue_size*self.num_experts / bs:
                self.if_init=True
                self.initialize_cluster_centers()
            loss_sim = 0.0
            dist_loss = 0.0
        else:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
                pseudo_labels_m = self.generate_pseudo_labels(features_m_flatten)
                #获取队列中的特征
                # 获取队列中的特征
                feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
                feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
                feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
                queue_count, queue_index = self.select_effective_features()

                # 将队列特征与伪标签组合
                feat1_one_hot = F.one_hot(torch.zeros(queue_count[0], dtype=torch.long), num_classes=self.num_experts).float().to(feat1_queue.device)
                feat2_one_hot = F.one_hot(torch.ones(queue_count[1], dtype=torch.long), num_classes=self.num_experts).float().to(feat2_queue.device)
                feat3_one_hot = F.one_hot(torch.full((queue_count[2],), 2, dtype=torch.long), num_classes=self.num_experts).float().to(feat3_queue.device)
                feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
                feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
                feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
                #features_m_flatten:[8, 384],feat1_effect_queue[384, 16]
                all_features = torch.cat([features_m_flatten.T, feat1_effect_queue, feat2_effect_queue, feat3_effect_queue], dim=1)
                pseudo_label_onehot_m = F.one_hot(pseudo_labels_m, num_classes=self.num_experts).float()
                all_labels = torch.cat([
                    pseudo_label_onehot_m,
                    feat1_one_hot,
                    feat2_one_hot,
                    feat3_one_hot
                ], dim=0)
                    # 生成伪标签
                temp = 1.0  # 温度参数
                sim_feat_m = features_m_flatten @ all_features / temp
                # 计算硬目标 hard_target
                hard_target = torch.mm(pseudo_label_onehot_m, all_labels.T) / temp
                sim_target = self.alpha * F.softmax(sim_feat_m) + (1- self.alpha)*hard_target

            sim_feat = features_flatten @ all_features / temp#
            density_weights = self.calculateDensityWeight(features_flatten)
            loss_sim = -torch.sum(F.log_softmax(sim_feat, dim=1) * sim_target, dim=1)
            loss_sim = torch.mul(density_weights, loss_sim).sum()
            # 更新队列和聚类中心
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.iteration_counter += 1
            if self.iteration_counter % self.update_frequency == 0:
                self.kmeans_iteration()
                self.iteration_counter = 0
            # 计算聚类中心距离损失，防止聚类坍塌
            dist_loss = self.cluster_distance_loss()
        #在queue没满之前我可以让损失都是0
        # 分类输出
        diversity_loss = self.calculate_diversity_loss(features_flatten)
        fuse_features = self.fuse_features(features, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        # final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        pseudo_labels_m_tensor = torch.tensor(pseudo_labels_m, dtype=torch.long).clone().detach().to(outputs_tensor.device)
        
        # 确保张量形状正确
        # adaptive_weight = self.gating(features_flatten.detach())
        # 计算监督损失
        # pseudo_loss = self.criterion(adaptive_weight, pseudo_labels_m_tensor)
        pseudo_labels_m_tensor = pseudo_labels_m_tensor.view(-1, 1, 1)  # 调整形状
        pseudo_labels_m_tensor = pseudo_labels_m_tensor.expand(-1, -1, outputs_tensor.shape[-1])  # 扩展形状
        # final_outputs = torch.einsum('bc,bcn->bcn',pseudo_labels_m_tensor, outputs_tensor)
        # final_outputs_summed = final_outputs.sum(dim=1)
        final_outputs = torch.gather(outputs_tensor, 1, pseudo_labels_m_tensor)
        final_outputs = final_outputs.squeeze(1)#直接使用伪标签的值
        return {
            'logits': torch.stack(outputs, dim=1),
            'output': final_outputs,
            'features': features_flatten,
            'pseudo_labels': pseudo_labels_m_tensor.squeeze(),
            'sim_loss': loss_sim,
            # 'pseudo_loss': pseudo_loss,
            'diversity_loss': diversity_loss,
            'distance_loss': dist_loss
        }
class ClinicalImageBaseClusterDistancePlusGatingModel(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModel, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.init_itr = 0
        self.k = k
        # 动量模型
        self.encoder_m = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.proj_m = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )

        self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        # 专家输出层
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        self.cluster_init_type = cluster_init_type
        self.gating = GatingNetwork(feat_dim,num_experts=num_experts)
        #多少次更新聚类中心
        # 定义损失函数，例如交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # 添加一个属性来跟踪迭代次数
        self.iteration_counter = 0
        self.num_iterations = num_iterations
        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.proj,self.proj_m],
                           ]
        self.copy_params()
        # 定义每多少次迭代更新一次聚类中心
        self.update_frequency = 100
        # 队列
        self.queue_size = queue_size
        self.n_clusters = num_experts
        
        self.register_buffer("feat_queues", torch.zeros(num_experts, feat_dim, queue_size))
        self.feat_queues = nn.functional.normalize(self.feat_queues, dim=1)
        self.register_buffer("queue_ptrs", torch.zeros(num_experts, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim))
        # self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim*num_experts))
        self.if_init = False
    def merge_feat_queues(self):
        """
        Merge all feature queues into a single tensor.
        """
        # merged_features = []
        feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
        feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
        feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
        queue_count, queue_index = self.select_effective_features()
        feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
        feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
        feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
        
        # 拼接所有特征，并沿第一维（样本数）拼接
        merged_features = torch.cat((feat1_effect_queue, feat2_effect_queue, feat3_effect_queue), dim=1)

        # 转置拼接后的特征矩阵
        return merged_features.t()
    def select_effective_features(self):
        non_zero_indices = []
        non_zero_counts = []
        for i in range(self.num_experts):
            # 检查feat_queues[i,k,:]是否全为0，使用torch.any结合非等于0的条件
            # 检查每一列是否全部为零
            non_zero = torch.any(self.feat_queues[i] != 0, dim=0)  # shape: (96,)

            # 获取有效的特征索引
            non_zero_indices_i = torch.nonzero(non_zero, as_tuple=True)[0]  # shape: (N,)
            non_zero_indices.append(non_zero_indices_i)
            non_zero_counts.append(len(non_zero_indices_i))
        return non_zero_counts, non_zero_indices
        # print("Non-zero feature indices:", non_zero_indices)
        # print("Count of non-zero features:", non_zero_counts)
    
    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    @torch.no_grad()
    def extract_features_m(self, image):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder_m(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj_m(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        return features_flatten, img_feature_list, 
    def fuse_features(self, features, text_features):
        #DAF模块
        index = generate_attention_sequence(self.num_experts, len(features))
        attentioned_features = []
        feat_last = features[-1]
        for idx in range(len(index)):
            att_feat = self.attention[idx](features[index[idx]], feat_last)
            attentioned_features.append(att_feat)
        img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        aligned_features = []
        text_features = text_features.view(text_features.size(0), -1)
        for idx in range(self.num_experts):
            aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
            aligned_features.append(aligned_feat)
        return aligned_features
    # def momentum_forward(self, image, clinical_data):
    #     return self.extract_and_align_features(image, clinical_data, self.encoder_m)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def update_queues(self, features, pseudo_labels):
        for i in range(self.num_experts):
            ptr = int(self.queue_ptrs[i].item())
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
            idx = (pseudo_labels_tensor == i).nonzero(as_tuple=True)[0]
            # feat_i = torch.gather(features, 1, pseudo_labels_tensor)#16,384
            # feat_i = features[pseudo_labels == i]#这个就是更新
            if len(idx) > 0:
                # 根据索引获取特征
                feat_i = features[idx]
            # if feat_i.size(0) > 0:
                # Update the queue with new features
                num_features = min(feat_i.size(0), self.queue_size - ptr)
                self.feat_queues[i, :, ptr:ptr + num_features] = feat_i[:num_features].t()
                # Move the pointer
                ptr = (ptr + num_features) % self.queue_size
                self.queue_ptrs[i] = ptr

    @torch.no_grad()
    def reassign_queue_features(self):
        # 遍历每个专家队列
        for i in range(self.num_experts):
            # 获取当前队列中的有效特征（非空部分）
            valid_idx = self.queue_ptrs[i].item() if self.queue_ptrs[i].item() > 0 else self.queue_size
            queue_features = self.feat_queues[i, :, :valid_idx].clone().detach().t()  # shape: (valid_size, feat_dim*num_experts)
            
            # 计算每个特征到所有聚类中心的距离
            distances = torch.cdist(queue_features, self.cluster_centers)
            
            # 找到每个特征最近的聚类中心
            new_pseudo_labels = torch.argmin(distances, dim=1)
            
            # 根据新的伪标签重新分配特征到新的队列
            for j in range(self.num_experts):
                # 找到需要分配到队列 j 的特征
                features_to_assign = queue_features[new_pseudo_labels == j]
                
                # 更新队列 j 中的特征
                ptr = int(self.queue_ptrs[j].item())
                num_features = min(features_to_assign.size(0), self.queue_size - ptr)
                
                if num_features > 0:
                    # 更新队列 j 中的特征
                    self.feat_queues[j, :, ptr:ptr + num_features] = features_to_assign[:num_features].t()
                    
                    # 更新指针位置，指向第一个空位置
                    self.queue_ptrs[j] = (ptr + num_features) % self.queue_size
                
                # 如果本次重新分配的特征数量不足以填满队列，则将指针设置为有效的最大位置
                if num_features < features_to_assign.size(0):
                    self.queue_ptrs[j] = self.queue_size

    def entropy_regularization(self, labels):
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        probabilities = cluster_counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    @torch.no_grad()
    def update_cluster_centers(self):
        # Update the cluster centers by averaging the features in each queue
        for i in range(self.num_experts):
            self.cluster_centers[i] = self.feat_queues[i].mean(dim=1).clone().detach()
            self.cluster_centers[i] = nn.functional.normalize(self.cluster_centers[i], dim=0)
    def initialize_cluster_centers(self):
        if self.cluster_init_type == "kmeans":
            self.update_cluster_centers()
            self.kmeans_iteration()
        elif self.cluster_init_type == "kmeans++":
            self.kmeans_plus_plus_init()
            self.kmeans_iteration()
    #使用feature向量的密度进行加权，首先要计算密度,防止聚类坍塌
    def calculateDensityWeight(self, features):
        #计算密度
        merge_feature = self.merge_feat_queues()
        distances = torch.cdist(features, merge_feature)

        # 找到每个输入特征最近的k个邻居的距离
        _, topk_indices = torch.topk(distances, k=self.k, largest=False, sorted=True)
        topk_distances = torch.gather(distances, 1, topk_indices)

        # 计算密度，这里简单地取距离的倒数作为密度的度量
        # 可以考虑对距离进行规范化或者使用其他方法来计算密度
        densities = 1.0 / (topk_distances.sum(dim=1) + 1e-8)  # 加上一个小数值防止除以零
        weights = F.softmax(-densities, dim=0)
        return weights
    def kmeans_plus_plus_init(self):
        """
        使用 KMeans++ 算法初始化聚类中心
        """
        # 合并特征队列
        merge_feature = self.merge_feat_queues()
        
        # 随机选择第一个聚类中心
        centers = [merge_feature[torch.randint(0, merge_feature.size(0), (1,)).item()]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到已选择聚类中心的最近距离
            # 使用torch.min和广播机制避免for循环，提高效率
            distances = torch.stack([torch.norm(merge_feature - c, dim=1).pow(2) for c in centers], dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离计算选择下一个中心的概率
            probabilities = min_distances / min_distances.sum()
            
            # 生成累积概率分布
            cumulative_probabilities = probabilities.cumsum(dim=0)
            
            # 按照概率选择下一个聚类中心
            r = torch.rand(1).item()
            i = torch.where(cumulative_probabilities >= r)[0][0].item()
            centers.append(merge_feature[i])

        # 将中心列表转换为torch张量并赋值给cluster_centers
        self.cluster_centers = torch.nn.Parameter(torch.stack(centers))

    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")

    def generate_pseudo_labels(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
        distances = torch.cdist(features, self.cluster_centers)
        pseudo_labels = torch.argmin(distances, dim=1)  # 选择最近的聚类中心作为标签
        return pseudo_labels
    def calculate_diversity_loss(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
         # 获取当前特征所在的设备
        device = features.device

        # 如果聚类中心在CPU上，将其移动到与features相同的设备
        cluster_centers_on_device = self.cluster_centers.to(device)

        # 计算样本到各个聚类中心的距离
        distances = torch.cdist(features, cluster_centers_on_device)

        # 归一化每个样本的距离
        normalized_distances = distances / distances.sum(dim=1, keepdim=True)

        # 计算方差
        variance = torch.var(normalized_distances, dim=1).mean()

        # 返回多样性损失（负方差）
        return -variance
    def cluster_distance_loss(self):
        # 防止聚类坍塌，最小化聚类中心之间的距离
        dist_loss = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                dist_loss += 1 / (dist + 1e-6)  # 防止除零
        return dist_loss
    def forward(self, image, text):
        # 提取特征并对齐
        features_flatten, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]
        # features = torch.stack(features, dim=1)#B,N,feature_dim

        # features_m = torch.stack(features_m, dim=1)
        if self.if_init == False:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
            bs = features_m_flatten.size(0)
            pseudo_labels_m = random.choices([0, 1, 2], k=bs)
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.init_itr +=1
            if self.init_itr > self.queue_size*self.num_experts / bs:
                self.if_init=True
                self.initialize_cluster_centers()
            loss_sim = 0.0
            dist_loss = 0.0
        else:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
                pseudo_labels_m = self.generate_pseudo_labels(features_m_flatten)
                #获取队列中的特征
                # 获取队列中的特征
                feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
                feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
                feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
                queue_count, queue_index = self.select_effective_features()

                # 将队列特征与伪标签组合
                feat1_one_hot = F.one_hot(torch.zeros(queue_count[0], dtype=torch.long), num_classes=self.num_experts).float().to(feat1_queue.device)
                feat2_one_hot = F.one_hot(torch.ones(queue_count[1], dtype=torch.long), num_classes=self.num_experts).float().to(feat2_queue.device)
                feat3_one_hot = F.one_hot(torch.full((queue_count[2],), 2, dtype=torch.long), num_classes=self.num_experts).float().to(feat3_queue.device)
                feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
                feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
                feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
                #features_m_flatten:[8, 384],feat1_effect_queue[384, 16]
                all_features = torch.cat([features_m_flatten.T, feat1_effect_queue, feat2_effect_queue, feat3_effect_queue], dim=1)
                pseudo_label_onehot_m = F.one_hot(pseudo_labels_m, num_classes=self.num_experts).float()
                all_labels = torch.cat([
                    pseudo_label_onehot_m,
                    feat1_one_hot,
                    feat2_one_hot,
                    feat3_one_hot
                ], dim=0)
                    # 生成伪标签
                temp = 1.0  # 温度参数
                sim_feat_m = features_m_flatten @ all_features / temp
                # 计算硬目标 hard_target
                hard_target = torch.mm(pseudo_label_onehot_m, all_labels.T) / temp
                sim_target = self.alpha * F.softmax(sim_feat_m) + (1- self.alpha)*hard_target

            sim_feat = features_flatten @ all_features / temp#
            density_weights = self.calculateDensityWeight(features_flatten)
            loss_sim = -torch.sum(F.log_softmax(sim_feat, dim=1) * sim_target, dim=1)
            loss_sim = torch.mul(density_weights, loss_sim).sum()
            # 更新队列和聚类中心
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.iteration_counter += 1
            if self.iteration_counter % self.update_frequency == 0:
                self.kmeans_iteration()
                self.iteration_counter = 0
            # 计算聚类中心距离损失，防止聚类坍塌
            dist_loss = self.cluster_distance_loss()
        #在queue没满之前我可以让损失都是0
        # 分类输出
        diversity_loss = self.calculate_diversity_loss(features_flatten)
        fuse_features = self.fuse_features(features, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        # final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        pseudo_labels_m_tensor = torch.tensor(pseudo_labels_m, dtype=torch.long).clone().detach().to(outputs_tensor.device)
        
        # 确保张量形状正确
        adaptive_weight = self.gating(features_flatten.detach())
        # 计算监督损失
        pseudo_loss = self.criterion(adaptive_weight, pseudo_labels_m_tensor)
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.view(-1, 1, 1)  # 调整形状
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.expand(-1, -1, outputs_tensor.shape[-1])  # 扩展形状
        final_outputs = torch.einsum('bc,bcn->bcn',adaptive_weight.detach(), outputs_tensor)
        final_outputs_summed = final_outputs.sum(dim=1)
        # final_outputs = torch.gather(outputs_tensor, 1, pseudo_labels_m_tensor)
        # final_outputs = final_outputs.squeeze(1)#直接使用伪标签的值
        return {
            'logits': torch.stack(outputs, dim=1),
            'output': final_outputs_summed,
            'features': features_flatten,
            'pseudo_labels': pseudo_labels_m_tensor.squeeze(),
            'sim_loss': loss_sim,
            'pseudo_loss': pseudo_loss,
            'diversity_loss': diversity_loss,
            'distance_loss': dist_loss
        }
class ClinicalImageBaseClusterDistancePlusGatingModelF(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelF, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.init_itr = 0
        self.k = k
        # 动量模型
        self.encoder_m = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        
        self.proj_m = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fuse_proj = nn.ModuleList([nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()) for _ in range(3)])

        # self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        # self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        # 专家输出层
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        self.cluster_init_type = cluster_init_type
        self.gating = GatingNetwork(feat_dim,num_experts=num_experts)
        #多少次更新聚类中心
        # 定义损失函数，例如交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # 添加一个属性来跟踪迭代次数
        self.iteration_counter = 0
        self.num_iterations = num_iterations
        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.proj,self.proj_m],
                           ]
        self.copy_params()
        # 定义每多少次迭代更新一次聚类中心
        self.update_frequency = 100
        # 队列
        self.queue_size = queue_size
        self.n_clusters = num_experts
        
        self.register_buffer("feat_queues", torch.zeros(num_experts, feat_dim, queue_size))
        self.feat_queues = nn.functional.normalize(self.feat_queues, dim=1)
        self.register_buffer("queue_ptrs", torch.zeros(num_experts, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim))
        # self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim*num_experts))
        self.if_init = False
    def merge_feat_queues(self):
        """
        Merge all feature queues into a single tensor.
        """
        # merged_features = []
        feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
        feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
        feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
        queue_count, queue_index = self.select_effective_features()
        feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
        feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
        feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
        
        # 拼接所有特征，并沿第一维（样本数）拼接
        merged_features = torch.cat((feat1_effect_queue, feat2_effect_queue, feat3_effect_queue), dim=1)

        # 转置拼接后的特征矩阵
        return merged_features.t()
    def select_effective_features(self):
        non_zero_indices = []
        non_zero_counts = []
        for i in range(self.num_experts):
            # 检查feat_queues[i,k,:]是否全为0，使用torch.any结合非等于0的条件
            # 检查每一列是否全部为零
            non_zero = torch.any(self.feat_queues[i] != 0, dim=0)  # shape: (96,)

            # 获取有效的特征索引
            non_zero_indices_i = torch.nonzero(non_zero, as_tuple=True)[0]  # shape: (N,)
            non_zero_indices.append(non_zero_indices_i)
            non_zero_counts.append(len(non_zero_indices_i))
        return non_zero_counts, non_zero_indices
        # print("Non-zero feature indices:", non_zero_indices)
        # print("Count of non-zero features:", non_zero_counts)
    
    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    @torch.no_grad()
    def extract_features_m(self, image):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder_m(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj_m(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        return features_flatten, img_feature_list, 
    
    def fuse_features(self, features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        aligned_features = []
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        cat_features = torch.cat((features, text_features), dim=1)
        for idx in range(self.num_experts):
            aligned_feat = self.fuse_proj[idx](cat_features)
            aligned_features.append(aligned_feat)

        return aligned_features
        # #DAF模块
        # index = generate_attention_sequence(self.num_experts, len(features))
        # attentioned_features = []
        # feat_last = features[-1]
        # # for idx in range(len(index)):
        # #     att_feat = self.attention[idx](features[index[idx]], feat_last)
        # #     attentioned_features.append(att_feat)
        # img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        # aligned_features = []
        # text_features = text_features.view(text_features.size(0), -1)
        # for idx in range(self.num_experts):
        #     aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
        #     aligned_features.append(aligned_feat)
        # return aligned_features
    # def momentum_forward(self, image, clinical_data):
    #     return self.extract_and_align_features(image, clinical_data, self.encoder_m)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def update_queues(self, features, pseudo_labels):
        for i in range(self.num_experts):
            ptr = int(self.queue_ptrs[i].item())
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
            idx = (pseudo_labels_tensor == i).nonzero(as_tuple=True)[0]
            # feat_i = torch.gather(features, 1, pseudo_labels_tensor)#16,384
            # feat_i = features[pseudo_labels == i]#这个就是更新
            if len(idx) > 0:
                # 根据索引获取特征
                feat_i = features[idx]
            # if feat_i.size(0) > 0:
                # Update the queue with new features
                num_features = min(feat_i.size(0), self.queue_size - ptr)
                self.feat_queues[i, :, ptr:ptr + num_features] = feat_i[:num_features].t()
                # Move the pointer
                ptr = (ptr + num_features) % self.queue_size
                self.queue_ptrs[i] = ptr

    @torch.no_grad()
    def reassign_queue_features(self):
        # 遍历每个专家队列
        for i in range(self.num_experts):
            # 获取当前队列中的有效特征（非空部分）
            valid_idx = self.queue_ptrs[i].item() if self.queue_ptrs[i].item() > 0 else self.queue_size
            queue_features = self.feat_queues[i, :, :valid_idx].clone().detach().t()  # shape: (valid_size, feat_dim*num_experts)
            
            # 计算每个特征到所有聚类中心的距离
            distances = torch.cdist(queue_features, self.cluster_centers)
            
            # 找到每个特征最近的聚类中心
            new_pseudo_labels = torch.argmin(distances, dim=1)
            
            # 根据新的伪标签重新分配特征到新的队列
            for j in range(self.num_experts):
                # 找到需要分配到队列 j 的特征
                features_to_assign = queue_features[new_pseudo_labels == j]
                
                # 更新队列 j 中的特征
                ptr = int(self.queue_ptrs[j].item())
                num_features = min(features_to_assign.size(0), self.queue_size - ptr)
                
                if num_features > 0:
                    # 更新队列 j 中的特征
                    self.feat_queues[j, :, ptr:ptr + num_features] = features_to_assign[:num_features].t()
                    
                    # 更新指针位置，指向第一个空位置
                    self.queue_ptrs[j] = (ptr + num_features) % self.queue_size
                
                # 如果本次重新分配的特征数量不足以填满队列，则将指针设置为有效的最大位置
                if num_features < features_to_assign.size(0):
                    self.queue_ptrs[j] = self.queue_size

    def entropy_regularization(self, labels):
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        probabilities = cluster_counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    @torch.no_grad()
    def update_cluster_centers(self):
        # Update the cluster centers by averaging the features in each queue
        for i in range(self.num_experts):
            self.cluster_centers[i] = self.feat_queues[i].mean(dim=1).clone().detach()
            self.cluster_centers[i] = nn.functional.normalize(self.cluster_centers[i], dim=0)
    def initialize_cluster_centers(self):
        if self.cluster_init_type == "kmeans":
            self.update_cluster_centers()
            self.kmeans_iteration()
        elif self.cluster_init_type == "kmeans++":
            self.kmeans_plus_plus_init()
            self.kmeans_iteration()
    #使用feature向量的密度进行加权，首先要计算密度,防止聚类坍塌
    def calculateDensityWeight(self, features):
        #计算密度
        merge_feature = self.merge_feat_queues()
        distances = torch.cdist(features, merge_feature)

        # 找到每个输入特征最近的k个邻居的距离
        _, topk_indices = torch.topk(distances, k=self.k, largest=False, sorted=True)
        topk_distances = torch.gather(distances, 1, topk_indices)

        # 计算密度，这里简单地取距离的倒数作为密度的度量
        # 可以考虑对距离进行规范化或者使用其他方法来计算密度
        densities = 1.0 / (topk_distances.sum(dim=1) + 1e-8)  # 加上一个小数值防止除以零
        weights = F.softmax(-densities, dim=0)
        return weights
    def kmeans_plus_plus_init(self):
        """
        使用 KMeans++ 算法初始化聚类中心
        """
        # 合并特征队列
        merge_feature = self.merge_feat_queues()
        
        # 随机选择第一个聚类中心
        centers = [merge_feature[torch.randint(0, merge_feature.size(0), (1,)).item()]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到已选择聚类中心的最近距离
            # 使用torch.min和广播机制避免for循环，提高效率
            distances = torch.stack([torch.norm(merge_feature - c, dim=1).pow(2) for c in centers], dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离计算选择下一个中心的概率
            probabilities = min_distances / min_distances.sum()
            
            # 生成累积概率分布
            cumulative_probabilities = probabilities.cumsum(dim=0)
            
            # 按照概率选择下一个聚类中心
            r = torch.rand(1).item()
            i = torch.where(cumulative_probabilities >= r)[0][0].item()
            centers.append(merge_feature[i])

        # 将中心列表转换为torch张量并赋值给cluster_centers
        self.cluster_centers = torch.nn.Parameter(torch.stack(centers))

    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")

    def generate_pseudo_labels(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
        distances = torch.cdist(features, self.cluster_centers)
        pseudo_labels = torch.argmin(distances, dim=1)  # 选择最近的聚类中心作为标签
        return pseudo_labels
    def calculate_diversity_loss(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
         # 获取当前特征所在的设备
        device = features.device

        # 如果聚类中心在CPU上，将其移动到与features相同的设备
        cluster_centers_on_device = self.cluster_centers.to(device)

        # 计算样本到各个聚类中心的距离
        distances = torch.cdist(features, cluster_centers_on_device)

        # 归一化每个样本的距离
        normalized_distances = distances / distances.sum(dim=1, keepdim=True)

        # 计算方差
        variance = torch.var(normalized_distances, dim=1).mean()

        # 返回多样性损失（负方差）
        return -variance
    def cluster_distance_loss(self):
        # 防止聚类坍塌，最小化聚类中心之间的距离
        dist_loss = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                dist_loss += 1 / (dist + 1e-6)  # 防止除零
        return dist_loss
    def forward(self, image, text):
        # 提取特征并对齐
        features_flatten, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]
        # features = torch.stack(features, dim=1)#B,N,feature_dim

        # features_m = torch.stack(features_m, dim=1)
        if self.if_init == False:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
            bs = features_m_flatten.size(0)
            pseudo_labels_m = random.choices([0, 1, 2], k=bs)
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.init_itr +=1
            if self.init_itr > self.queue_size*self.num_experts / bs:
                self.if_init=True
                self.initialize_cluster_centers()
            loss_sim = 0.0
            dist_loss = 0.0
        else:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
                pseudo_labels_m = self.generate_pseudo_labels(features_m_flatten)
                #获取队列中的特征
                # 获取队列中的特征
                feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
                feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
                feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
                queue_count, queue_index = self.select_effective_features()

                # 将队列特征与伪标签组合
                feat1_one_hot = F.one_hot(torch.zeros(queue_count[0], dtype=torch.long), num_classes=self.num_experts).float().to(feat1_queue.device)
                feat2_one_hot = F.one_hot(torch.ones(queue_count[1], dtype=torch.long), num_classes=self.num_experts).float().to(feat2_queue.device)
                feat3_one_hot = F.one_hot(torch.full((queue_count[2],), 2, dtype=torch.long), num_classes=self.num_experts).float().to(feat3_queue.device)
                feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
                feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
                feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
                #features_m_flatten:[8, 384],feat1_effect_queue[384, 16]
                all_features = torch.cat([features_m_flatten.T, feat1_effect_queue, feat2_effect_queue, feat3_effect_queue], dim=1)
                pseudo_label_onehot_m = F.one_hot(pseudo_labels_m, num_classes=self.num_experts).float()
                all_labels = torch.cat([
                    pseudo_label_onehot_m,
                    feat1_one_hot,
                    feat2_one_hot,
                    feat3_one_hot
                ], dim=0)
                    # 生成伪标签
                temp = 1.0  # 温度参数
                sim_feat_m = features_m_flatten @ all_features / temp
                # 计算硬目标 hard_target
                hard_target = torch.mm(pseudo_label_onehot_m, all_labels.T) / temp
                sim_target = self.alpha * F.softmax(sim_feat_m) + (1- self.alpha)*hard_target

            sim_feat = features_flatten @ all_features / temp#
            density_weights = self.calculateDensityWeight(features_flatten)
            loss_sim = -torch.sum(F.log_softmax(sim_feat, dim=1) * sim_target, dim=1)
            loss_sim = torch.mul(density_weights, loss_sim).sum()
            # 更新队列和聚类中心
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.iteration_counter += 1
            if self.iteration_counter % self.update_frequency == 0:
                self.kmeans_iteration()
                self.iteration_counter = 0
            # 计算聚类中心距离损失，防止聚类坍塌
            dist_loss = self.cluster_distance_loss()
        #在queue没满之前我可以让损失都是0
        # 分类输出
        diversity_loss = self.calculate_diversity_loss(features_flatten)
        fuse_features = self.fuse_features(features_flatten, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        # final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        pseudo_labels_m_tensor = torch.tensor(pseudo_labels_m, dtype=torch.long).clone().detach().to(outputs_tensor.device)
        
        # 确保张量形状正确
        adaptive_weight = self.gating(features_flatten.detach())
        # 计算监督损失
        pseudo_loss = self.criterion(adaptive_weight, pseudo_labels_m_tensor)
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.view(-1, 1, 1)  # 调整形状
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.expand(-1, -1, outputs_tensor.shape[-1])  # 扩展形状
        final_outputs = torch.einsum('bc,bcn->bcn',adaptive_weight.detach(), outputs_tensor)
        final_outputs_summed = final_outputs.sum(dim=1)
        # final_outputs = torch.gather(outputs_tensor, 1, pseudo_labels_m_tensor)
        # final_outputs = final_outputs.squeeze(1)#直接使用伪标签的值
        return {
            'logits': torch.stack(outputs, dim=1),
            'output': final_outputs_summed,
            'features': features_flatten,
            'pseudo_labels': pseudo_labels_m_tensor.squeeze(),
            'sim_loss': loss_sim,
            'pseudo_loss': pseudo_loss,
            'diversity_loss': diversity_loss,
            'distance_loss': dist_loss
        }

class ClinicalImageBaseClusterDistancePlusGatingModelG(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelG, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.init_itr = 0
        self.k = k
        # 动量模型
        self.encoder_m = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        
        self.proj_m = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fuse_proj = nn.ModuleList([nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()) for _ in range(3)])

        self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        # self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        # 专家输出层
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        self.cluster_init_type = cluster_init_type
        self.gating = GatingNetwork(feat_dim,num_experts=num_experts)
        #多少次更新聚类中心
        # 定义损失函数，例如交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # 添加一个属性来跟踪迭代次数
        self.iteration_counter = 0
        self.num_iterations = num_iterations
        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.proj,self.proj_m],
                           ]
        self.copy_params()
        # 定义每多少次迭代更新一次聚类中心
        self.update_frequency = 100
        # 队列
        self.queue_size = queue_size
        self.n_clusters = num_experts
        
        self.register_buffer("feat_queues", torch.zeros(num_experts, feat_dim, queue_size))
        self.feat_queues = nn.functional.normalize(self.feat_queues, dim=1)
        self.register_buffer("queue_ptrs", torch.zeros(num_experts, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim))
        # self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim*num_experts))
        self.if_init = False
    def merge_feat_queues(self):
        """
        Merge all feature queues into a single tensor.
        """
        # merged_features = []
        feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
        feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
        feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
        queue_count, queue_index = self.select_effective_features()
        feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
        feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
        feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
        
        # 拼接所有特征，并沿第一维（样本数）拼接
        merged_features = torch.cat((feat1_effect_queue, feat2_effect_queue, feat3_effect_queue), dim=1)

        # 转置拼接后的特征矩阵
        return merged_features.t()
    def select_effective_features(self):
        non_zero_indices = []
        non_zero_counts = []
        for i in range(self.num_experts):
            # 检查feat_queues[i,k,:]是否全为0，使用torch.any结合非等于0的条件
            # 检查每一列是否全部为零
            non_zero = torch.any(self.feat_queues[i] != 0, dim=0)  # shape: (96,)

            # 获取有效的特征索引
            non_zero_indices_i = torch.nonzero(non_zero, as_tuple=True)[0]  # shape: (N,)
            non_zero_indices.append(non_zero_indices_i)
            non_zero_counts.append(len(non_zero_indices_i))
        return non_zero_counts, non_zero_indices
        # print("Non-zero feature indices:", non_zero_indices)
        # print("Count of non-zero features:", non_zero_counts)
    
    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    @torch.no_grad()
    def extract_features_m(self, image):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder_m(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj_m(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        return features_flatten, img_feature_list, 
    
    def fuse_features(self, features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        # assert features[0].size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        aligned_features = []
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        attentioned_features = []
        feat_last = features[-1]
        index = generate_attention_sequence(self.num_experts, len(features))
        for idx in range(len(index)):
            att_feat = self.attention[idx](features[index[idx]], feat_last)
            attentioned_features.append(att_feat)
        img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        cat_features = [torch.cat((img_feat, text_features), dim=1) for img_feat in img_features_list]
        for idx in range(self.num_experts):
            aligned_feat = self.fuse_proj[idx](cat_features[idx])
            aligned_features.append(aligned_feat)

        return aligned_features
        # #DAF模块
        # index = generate_attention_sequence(self.num_experts, len(features))
        # attentioned_features = []
        # feat_last = features[-1]
        # # for idx in range(len(index)):
        # #     att_feat = self.attention[idx](features[index[idx]], feat_last)
        # #     attentioned_features.append(att_feat)
        # img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        # aligned_features = []
        # text_features = text_features.view(text_features.size(0), -1)
        # for idx in range(self.num_experts):
        #     aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
        #     aligned_features.append(aligned_feat)
        # return aligned_features
    # def momentum_forward(self, image, clinical_data):
    #     return self.extract_and_align_features(image, clinical_data, self.encoder_m)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def update_queues(self, features, pseudo_labels):
        for i in range(self.num_experts):
            ptr = int(self.queue_ptrs[i].item())
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
            idx = (pseudo_labels_tensor == i).nonzero(as_tuple=True)[0]
            # feat_i = torch.gather(features, 1, pseudo_labels_tensor)#16,384
            # feat_i = features[pseudo_labels == i]#这个就是更新
            if len(idx) > 0:
                # 根据索引获取特征
                feat_i = features[idx]
            # if feat_i.size(0) > 0:
                # Update the queue with new features
                num_features = min(feat_i.size(0), self.queue_size - ptr)
                self.feat_queues[i, :, ptr:ptr + num_features] = feat_i[:num_features].t()
                # Move the pointer
                ptr = (ptr + num_features) % self.queue_size
                self.queue_ptrs[i] = ptr

    @torch.no_grad()
    def reassign_queue_features(self):
        # 遍历每个专家队列
        for i in range(self.num_experts):
            # 获取当前队列中的有效特征（非空部分）
            valid_idx = self.queue_ptrs[i].item() if self.queue_ptrs[i].item() > 0 else self.queue_size
            queue_features = self.feat_queues[i, :, :valid_idx].clone().detach().t()  # shape: (valid_size, feat_dim*num_experts)
            
            # 计算每个特征到所有聚类中心的距离
            distances = torch.cdist(queue_features, self.cluster_centers)
            
            # 找到每个特征最近的聚类中心
            new_pseudo_labels = torch.argmin(distances, dim=1)
            
            # 根据新的伪标签重新分配特征到新的队列
            for j in range(self.num_experts):
                # 找到需要分配到队列 j 的特征
                features_to_assign = queue_features[new_pseudo_labels == j]
                
                # 更新队列 j 中的特征
                ptr = int(self.queue_ptrs[j].item())
                num_features = min(features_to_assign.size(0), self.queue_size - ptr)
                
                if num_features > 0:
                    # 更新队列 j 中的特征
                    self.feat_queues[j, :, ptr:ptr + num_features] = features_to_assign[:num_features].t()
                    
                    # 更新指针位置，指向第一个空位置
                    self.queue_ptrs[j] = (ptr + num_features) % self.queue_size
                
                # 如果本次重新分配的特征数量不足以填满队列，则将指针设置为有效的最大位置
                if num_features < features_to_assign.size(0):
                    self.queue_ptrs[j] = self.queue_size

    def entropy_regularization(self, labels):
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        probabilities = cluster_counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    @torch.no_grad()
    def update_cluster_centers(self):
        # Update the cluster centers by averaging the features in each queue
        for i in range(self.num_experts):
            self.cluster_centers[i] = self.feat_queues[i].mean(dim=1).clone().detach()
            self.cluster_centers[i] = nn.functional.normalize(self.cluster_centers[i], dim=0)
    def initialize_cluster_centers(self):
        if self.cluster_init_type == "kmeans":
            self.update_cluster_centers()
            self.kmeans_iteration()
        elif self.cluster_init_type == "kmeans++":
            self.kmeans_plus_plus_init()
            self.kmeans_iteration()
    #使用feature向量的密度进行加权，首先要计算密度,防止聚类坍塌
    def calculateDensityWeight(self, features):
        #计算密度
        merge_feature = self.merge_feat_queues()
        distances = torch.cdist(features, merge_feature)

        # 找到每个输入特征最近的k个邻居的距离
        _, topk_indices = torch.topk(distances, k=self.k, largest=False, sorted=True)
        topk_distances = torch.gather(distances, 1, topk_indices)

        # 计算密度，这里简单地取距离的倒数作为密度的度量
        # 可以考虑对距离进行规范化或者使用其他方法来计算密度
        densities = 1.0 / (topk_distances.sum(dim=1) + 1e-8)  # 加上一个小数值防止除以零
        weights = F.softmax(-densities, dim=0)
        return weights
    def kmeans_plus_plus_init(self):
        """
        使用 KMeans++ 算法初始化聚类中心
        """
        # 合并特征队列
        merge_feature = self.merge_feat_queues()
        
        # 随机选择第一个聚类中心
        centers = [merge_feature[torch.randint(0, merge_feature.size(0), (1,)).item()]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到已选择聚类中心的最近距离
            # 使用torch.min和广播机制避免for循环，提高效率
            distances = torch.stack([torch.norm(merge_feature - c, dim=1).pow(2) for c in centers], dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离计算选择下一个中心的概率
            probabilities = min_distances / min_distances.sum()
            
            # 生成累积概率分布
            cumulative_probabilities = probabilities.cumsum(dim=0)
            
            # 按照概率选择下一个聚类中心
            r = torch.rand(1).item()
            i = torch.where(cumulative_probabilities >= r)[0][0].item()
            centers.append(merge_feature[i])

        # 将中心列表转换为torch张量并赋值给cluster_centers
        self.cluster_centers = torch.nn.Parameter(torch.stack(centers))

    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")

    def generate_pseudo_labels(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
        distances = torch.cdist(features, self.cluster_centers)
        pseudo_labels = torch.argmin(distances, dim=1)  # 选择最近的聚类中心作为标签
        return pseudo_labels
    def calculate_diversity_loss(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
         # 获取当前特征所在的设备
        device = features.device

        # 如果聚类中心在CPU上，将其移动到与features相同的设备
        cluster_centers_on_device = self.cluster_centers.to(device)

        # 计算样本到各个聚类中心的距离
        distances = torch.cdist(features, cluster_centers_on_device)

        # 归一化每个样本的距离
        normalized_distances = distances / distances.sum(dim=1, keepdim=True)

        # 计算方差
        variance = torch.var(normalized_distances, dim=1).mean()

        # 返回多样性损失（负方差）
        return -variance
    def cluster_distance_loss(self):
        # 防止聚类坍塌，最小化聚类中心之间的距离
        dist_loss = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                dist_loss += 1 / (dist + 1e-6)  # 防止除零
        return dist_loss
    def forward(self, image, text):
        # 提取特征并对齐
        features_flatten, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]
        # features = torch.stack(features, dim=1)#B,N,feature_dim

        # features_m = torch.stack(features_m, dim=1)
        if self.if_init == False:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
            bs = features_m_flatten.size(0)
            pseudo_labels_m = random.choices([0, 1, 2], k=bs)
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.init_itr +=1
            if self.init_itr > self.queue_size*self.num_experts / bs:
                self.if_init=True
                self.initialize_cluster_centers()
            loss_sim = 0.0
            dist_loss = 0.0
        else:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
                pseudo_labels_m = self.generate_pseudo_labels(features_m_flatten)
                #获取队列中的特征
                # 获取队列中的特征
                feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
                feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
                feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
                queue_count, queue_index = self.select_effective_features()

                # 将队列特征与伪标签组合
                feat1_one_hot = F.one_hot(torch.zeros(queue_count[0], dtype=torch.long), num_classes=self.num_experts).float().to(feat1_queue.device)
                feat2_one_hot = F.one_hot(torch.ones(queue_count[1], dtype=torch.long), num_classes=self.num_experts).float().to(feat2_queue.device)
                feat3_one_hot = F.one_hot(torch.full((queue_count[2],), 2, dtype=torch.long), num_classes=self.num_experts).float().to(feat3_queue.device)
                feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
                feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
                feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
                #features_m_flatten:[8, 384],feat1_effect_queue[384, 16]
                all_features = torch.cat([features_m_flatten.T, feat1_effect_queue, feat2_effect_queue, feat3_effect_queue], dim=1)
                pseudo_label_onehot_m = F.one_hot(pseudo_labels_m, num_classes=self.num_experts).float()
                all_labels = torch.cat([
                    pseudo_label_onehot_m,
                    feat1_one_hot,
                    feat2_one_hot,
                    feat3_one_hot
                ], dim=0)
                    # 生成伪标签
                temp = 1.0  # 温度参数
                sim_feat_m = features_m_flatten @ all_features / temp
                # 计算硬目标 hard_target
                hard_target = torch.mm(pseudo_label_onehot_m, all_labels.T) / temp
                sim_target = self.alpha * F.softmax(sim_feat_m) + (1- self.alpha)*hard_target

            sim_feat = features_flatten @ all_features / temp#
            density_weights = self.calculateDensityWeight(features_flatten)
            loss_sim = -torch.sum(F.log_softmax(sim_feat, dim=1) * sim_target, dim=1)
            loss_sim = torch.mul(density_weights, loss_sim).sum()
            # 更新队列和聚类中心
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.iteration_counter += 1
            if self.iteration_counter % self.update_frequency == 0:
                self.kmeans_iteration()
                self.iteration_counter = 0
            # 计算聚类中心距离损失，防止聚类坍塌
            dist_loss = self.cluster_distance_loss()
        #在queue没满之前我可以让损失都是0
        # 分类输出
        diversity_loss = self.calculate_diversity_loss(features_flatten)
        fuse_features = self.fuse_features(features, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        # final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        pseudo_labels_m_tensor = torch.tensor(pseudo_labels_m, dtype=torch.long).clone().detach().to(outputs_tensor.device)
        
        # 确保张量形状正确
        adaptive_weight = self.gating(features_flatten.detach())
        # 计算监督损失
        pseudo_loss = self.criterion(adaptive_weight, pseudo_labels_m_tensor)
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.view(-1, 1, 1)  # 调整形状
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.expand(-1, -1, outputs_tensor.shape[-1])  # 扩展形状
        final_outputs = torch.einsum('bc,bcn->bcn',adaptive_weight.detach(), outputs_tensor)
        final_outputs_summed = final_outputs.sum(dim=1)
        # final_outputs = torch.gather(outputs_tensor, 1, pseudo_labels_m_tensor)
        # final_outputs = final_outputs.squeeze(1)#直接使用伪标签的值
        return {
            'logits': torch.stack(outputs, dim=1),
            'output': final_outputs_summed,
            'features': features_flatten,
            'pseudo_labels': pseudo_labels_m_tensor.squeeze(),
            'sim_loss': loss_sim,
            'pseudo_loss': pseudo_loss,
            'diversity_loss': diversity_loss,
            'distance_loss': dist_loss
        }
class ClinicalImageBaseClusterDistancePlusGatingModelH(nn.Module):
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelH, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages
        index = generate_attention_sequence(self.num_experts, len(self.stages)-1)
        self.init_itr = 0
        self.k = k
        # 动量模型
        self.encoder_m = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        
        self.proj_m = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fuse_proj = nn.ModuleList([nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()) for _ in range(3)])

        # self.attention = nn.ModuleList([Attention_block(self.stages[idx+1], self.stages[-1], 128) for idx in index])
        self.cross_attention = nn.ModuleList([CrossModalAttention(128, 64, feat_dim) for _ in range(num_experts)])
        # 专家输出层
        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        self.cluster_init_type = cluster_init_type
        self.gating = GatingNetwork(feat_dim,num_experts=num_experts)
        #多少次更新聚类中心
        # 定义损失函数，例如交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # 添加一个属性来跟踪迭代次数
        self.iteration_counter = 0
        self.num_iterations = num_iterations
        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.proj,self.proj_m],
                           ]
        self.copy_params()
        # 定义每多少次迭代更新一次聚类中心
        self.update_frequency = 100
        # 队列
        self.queue_size = queue_size
        self.n_clusters = num_experts
        
        self.register_buffer("feat_queues", torch.zeros(num_experts, feat_dim, queue_size))
        self.feat_queues = nn.functional.normalize(self.feat_queues, dim=1)
        self.register_buffer("queue_ptrs", torch.zeros(num_experts, dtype=torch.long))
        self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim))
        # self.cluster_centers = nn.Parameter(torch.zeros(num_experts, feat_dim*num_experts))
        self.if_init = False
    def merge_feat_queues(self):
        """
        Merge all feature queues into a single tensor.
        """
        # merged_features = []
        feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
        feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
        feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
        queue_count, queue_index = self.select_effective_features()
        feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
        feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
        feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
        
        # 拼接所有特征，并沿第一维（样本数）拼接
        merged_features = torch.cat((feat1_effect_queue, feat2_effect_queue, feat3_effect_queue), dim=1)

        # 转置拼接后的特征矩阵
        return merged_features.t()
    def select_effective_features(self):
        non_zero_indices = []
        non_zero_counts = []
        for i in range(self.num_experts):
            # 检查feat_queues[i,k,:]是否全为0，使用torch.any结合非等于0的条件
            # 检查每一列是否全部为零
            non_zero = torch.any(self.feat_queues[i] != 0, dim=0)  # shape: (96,)

            # 获取有效的特征索引
            non_zero_indices_i = torch.nonzero(non_zero, as_tuple=True)[0]  # shape: (N,)
            non_zero_indices.append(non_zero_indices_i)
            non_zero_counts.append(len(non_zero_indices_i))
        return non_zero_counts, non_zero_indices
        # print("Non-zero feature indices:", non_zero_indices)
        # print("Count of non-zero features:", non_zero_counts)
    
    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    @torch.no_grad()
    def extract_features_m(self, image):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder_m(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj_m(feat_last)
        features_flatten = nn.functional.normalize(features_flatten, dim=1)
        return features_flatten, img_feature_list, 
    
    def fuse_features(self, features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        # assert features[0].size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        aligned_features = []
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        # attentioned_features = []
        # feat_last = features[-1]
        # index = generate_attention_sequence(self.num_experts, len(features))
        # for idx in range(len(index)):
        #     att_feat = self.attention[idx](features[index[idx]], feat_last)
        #     attentioned_features.append(att_feat)
        # img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        # cat_features = [torch.cat((img_feat, text_features), dim=1) for img_feat in img_features_list]
        # for idx in range(self.num_experts):
        #     aligned_feat = self.fuse_proj[idx](cat_features[idx])
        #     aligned_features.append(aligned_feat)
        for idx in range(self.num_experts):
            aligned_feat = self.cross_attention[idx](features, text_features)
            aligned_features.append(aligned_feat)
        return aligned_features
        # #DAF模块
        # index = generate_attention_sequence(self.num_experts, len(features))
        # attentioned_features = []
        # feat_last = features[-1]
        # # for idx in range(len(index)):
        # #     att_feat = self.attention[idx](features[index[idx]], feat_last)
        # #     attentioned_features.append(att_feat)
        # img_features_list = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features]
        # aligned_features = []
        # text_features = text_features.view(text_features.size(0), -1)
        # for idx in range(self.num_experts):
        #     aligned_feat = self.cross_attention[idx](img_features_list[idx], text_features)
        #     aligned_features.append(aligned_feat)
        # return aligned_features
    # def momentum_forward(self, image, clinical_data):
    #     return self.extract_and_align_features(image, clinical_data, self.encoder_m)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def update_queues(self, features, pseudo_labels):
        for i in range(self.num_experts):
            ptr = int(self.queue_ptrs[i].item())
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
            idx = (pseudo_labels_tensor == i).nonzero(as_tuple=True)[0]
            # feat_i = torch.gather(features, 1, pseudo_labels_tensor)#16,384
            # feat_i = features[pseudo_labels == i]#这个就是更新
            if len(idx) > 0:
                # 根据索引获取特征
                feat_i = features[idx]
            # if feat_i.size(0) > 0:
                # Update the queue with new features
                num_features = min(feat_i.size(0), self.queue_size - ptr)
                self.feat_queues[i, :, ptr:ptr + num_features] = feat_i[:num_features].t()
                # Move the pointer
                ptr = (ptr + num_features) % self.queue_size
                self.queue_ptrs[i] = ptr

    @torch.no_grad()
    def reassign_queue_features(self):
        # 遍历每个专家队列
        for i in range(self.num_experts):
            # 获取当前队列中的有效特征（非空部分）
            valid_idx = self.queue_ptrs[i].item() if self.queue_ptrs[i].item() > 0 else self.queue_size
            queue_features = self.feat_queues[i, :, :valid_idx].clone().detach().t()  # shape: (valid_size, feat_dim*num_experts)
            
            # 计算每个特征到所有聚类中心的距离
            distances = torch.cdist(queue_features, self.cluster_centers)
            
            # 找到每个特征最近的聚类中心
            new_pseudo_labels = torch.argmin(distances, dim=1)
            
            # 根据新的伪标签重新分配特征到新的队列
            for j in range(self.num_experts):
                # 找到需要分配到队列 j 的特征
                features_to_assign = queue_features[new_pseudo_labels == j]
                
                # 更新队列 j 中的特征
                ptr = int(self.queue_ptrs[j].item())
                num_features = min(features_to_assign.size(0), self.queue_size - ptr)
                
                if num_features > 0:
                    # 更新队列 j 中的特征
                    self.feat_queues[j, :, ptr:ptr + num_features] = features_to_assign[:num_features].t()
                    
                    # 更新指针位置，指向第一个空位置
                    self.queue_ptrs[j] = (ptr + num_features) % self.queue_size
                
                # 如果本次重新分配的特征数量不足以填满队列，则将指针设置为有效的最大位置
                if num_features < features_to_assign.size(0):
                    self.queue_ptrs[j] = self.queue_size

    def entropy_regularization(self, labels):
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        probabilities = cluster_counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    @torch.no_grad()
    def update_cluster_centers(self):
        # Update the cluster centers by averaging the features in each queue
        for i in range(self.num_experts):
            self.cluster_centers[i] = self.feat_queues[i].mean(dim=1).clone().detach()
            self.cluster_centers[i] = nn.functional.normalize(self.cluster_centers[i], dim=0)
    def initialize_cluster_centers(self):
        if self.cluster_init_type == "kmeans":
            self.update_cluster_centers()
            self.kmeans_iteration()
        elif self.cluster_init_type == "kmeans++":
            self.kmeans_plus_plus_init()
            self.kmeans_iteration()
    #使用feature向量的密度进行加权，首先要计算密度,防止聚类坍塌
    def calculateDensityWeight(self, features):
        #计算密度
        merge_feature = self.merge_feat_queues()
        distances = torch.cdist(features, merge_feature)

        # 找到每个输入特征最近的k个邻居的距离
        _, topk_indices = torch.topk(distances, k=self.k, largest=False, sorted=True)
        topk_distances = torch.gather(distances, 1, topk_indices)

        # 计算密度，这里简单地取距离的倒数作为密度的度量
        # 可以考虑对距离进行规范化或者使用其他方法来计算密度
        densities = 1.0 / (topk_distances.sum(dim=1) + 1e-8)  # 加上一个小数值防止除以零
        weights = F.softmax(-densities, dim=0)
        return weights
    def kmeans_plus_plus_init(self):
        """
        使用 KMeans++ 算法初始化聚类中心
        """
        # 合并特征队列
        merge_feature = self.merge_feat_queues()
        
        # 随机选择第一个聚类中心
        centers = [merge_feature[torch.randint(0, merge_feature.size(0), (1,)).item()]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到已选择聚类中心的最近距离
            # 使用torch.min和广播机制避免for循环，提高效率
            distances = torch.stack([torch.norm(merge_feature - c, dim=1).pow(2) for c in centers], dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离计算选择下一个中心的概率
            probabilities = min_distances / min_distances.sum()
            
            # 生成累积概率分布
            cumulative_probabilities = probabilities.cumsum(dim=0)
            
            # 按照概率选择下一个聚类中心
            r = torch.rand(1).item()
            i = torch.where(cumulative_probabilities >= r)[0][0].item()
            centers.append(merge_feature[i])

        # 将中心列表转换为torch张量并赋值给cluster_centers
        self.cluster_centers = torch.nn.Parameter(torch.stack(centers))

    def kmeans_iteration(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration+1}/{self.num_iterations}: Updating cluster centers and reassigning features")
            # 更新聚类中心
            self.update_cluster_centers()

            # 重新分配特征到对应的聚类中心
            self.reassign_queue_features()

        print("KMeans iterations complete.")

    def generate_pseudo_labels(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
        distances = torch.cdist(features, self.cluster_centers)
        pseudo_labels = torch.argmin(distances, dim=1)  # 选择最近的聚类中心作为标签
        return pseudo_labels
    def calculate_diversity_loss(self, features):
        """
        使用当前聚类中心计算样本到各个聚类中心的距离，并分配伪标签
        """
         # 获取当前特征所在的设备
        device = features.device

        # 如果聚类中心在CPU上，将其移动到与features相同的设备
        cluster_centers_on_device = self.cluster_centers.to(device)

        # 计算样本到各个聚类中心的距离
        distances = torch.cdist(features, cluster_centers_on_device)

        # 归一化每个样本的距离
        normalized_distances = distances / distances.sum(dim=1, keepdim=True)

        # 计算方差
        variance = torch.var(normalized_distances, dim=1).mean()

        # 返回多样性损失（负方差）
        return -variance
    def cluster_distance_loss(self):
        # 防止聚类坍塌，最小化聚类中心之间的距离
        dist_loss = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                dist = torch.norm(self.cluster_centers[i] - self.cluster_centers[j])
                dist_loss += 1 / (dist + 1e-6)  # 防止除零
        return dist_loss
    def forward(self, image, text):
        # 提取特征并对齐
        features_flatten, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]
        # features = torch.stack(features, dim=1)#B,N,feature_dim

        # features_m = torch.stack(features_m, dim=1)
        if self.if_init == False:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
            bs = features_m_flatten.size(0)
            pseudo_labels_m = random.choices([0, 1, 2], k=bs)
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.init_itr +=1
            if self.init_itr > self.queue_size*self.num_experts / bs:
                self.if_init=True
                self.initialize_cluster_centers()
            loss_sim = 0.0
            dist_loss = 0.0
        else:
            with torch.no_grad():
                self._momentum_update()
                features_m_flatten, _ = self.extract_features_m(image)
                pseudo_labels_m = self.generate_pseudo_labels(features_m_flatten)
                #获取队列中的特征
                # 获取队列中的特征
                feat1_queue = self.feat_queues[0]  # 假设第一个专家的队列特征
                feat2_queue = self.feat_queues[1]  # 假设第二个专家的队列特征
                feat3_queue = self.feat_queues[2]  # 假设第三个专家的队列特征
                queue_count, queue_index = self.select_effective_features()

                # 将队列特征与伪标签组合
                feat1_one_hot = F.one_hot(torch.zeros(queue_count[0], dtype=torch.long), num_classes=self.num_experts).float().to(feat1_queue.device)
                feat2_one_hot = F.one_hot(torch.ones(queue_count[1], dtype=torch.long), num_classes=self.num_experts).float().to(feat2_queue.device)
                feat3_one_hot = F.one_hot(torch.full((queue_count[2],), 2, dtype=torch.long), num_classes=self.num_experts).float().to(feat3_queue.device)
                feat1_effect_queue = feat1_queue.clone().detach()[:,queue_index[0]]
                feat2_effect_queue = feat2_queue.clone().detach()[:,queue_index[1]]
                feat3_effect_queue = feat3_queue.clone().detach()[:,queue_index[2]]
                #features_m_flatten:[8, 384],feat1_effect_queue[384, 16]
                all_features = torch.cat([features_m_flatten.T, feat1_effect_queue, feat2_effect_queue, feat3_effect_queue], dim=1)
                pseudo_label_onehot_m = F.one_hot(pseudo_labels_m, num_classes=self.num_experts).float()
                all_labels = torch.cat([
                    pseudo_label_onehot_m,
                    feat1_one_hot,
                    feat2_one_hot,
                    feat3_one_hot
                ], dim=0)
                    # 生成伪标签
                temp = 1.0  # 温度参数
                sim_feat_m = features_m_flatten @ all_features / temp
                # 计算硬目标 hard_target
                hard_target = torch.mm(pseudo_label_onehot_m, all_labels.T) / temp
                sim_target = self.alpha * F.softmax(sim_feat_m) + (1- self.alpha)*hard_target

            sim_feat = features_flatten @ all_features / temp#
            density_weights = self.calculateDensityWeight(features_flatten)
            loss_sim = -torch.sum(F.log_softmax(sim_feat, dim=1) * sim_target, dim=1)
            loss_sim = torch.mul(density_weights, loss_sim).sum()
            # 更新队列和聚类中心
            self.update_queues(features_m_flatten, pseudo_labels_m)
            self.iteration_counter += 1
            if self.iteration_counter % self.update_frequency == 0:
                self.kmeans_iteration()
                self.iteration_counter = 0
            # 计算聚类中心距离损失，防止聚类坍塌
            dist_loss = self.cluster_distance_loss()
        #在queue没满之前我可以让损失都是0
        # 分类输出
        diversity_loss = self.calculate_diversity_loss(features_flatten)
        fuse_features = self.fuse_features(features_flatten, text_feat)
        outputs = [self.fusion_fc_list[idx](fuse_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        # final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        pseudo_labels_m_tensor = torch.tensor(pseudo_labels_m, dtype=torch.long).clone().detach().to(outputs_tensor.device)
        
        # 确保张量形状正确
        adaptive_weight = self.gating(features_flatten.detach())
        # 计算监督损失
        pseudo_loss = self.criterion(adaptive_weight, pseudo_labels_m_tensor)
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.view(-1, 1, 1)  # 调整形状
        # pseudo_labels_m_tensor = pseudo_labels_m_tensor.expand(-1, -1, outputs_tensor.shape[-1])  # 扩展形状
        final_outputs = torch.einsum('bc,bcn->bcn',adaptive_weight.detach(), outputs_tensor)
        final_outputs_summed = final_outputs.sum(dim=1)
        # final_outputs = torch.gather(outputs_tensor, 1, pseudo_labels_m_tensor)
        # final_outputs = final_outputs.squeeze(1)#直接使用伪标签的值
        return {
            'logits': torch.stack(outputs, dim=1),
            'output': final_outputs_summed,
            'features': features_flatten,
            'pseudo_labels': pseudo_labels_m_tensor.squeeze(),
            'sim_loss': loss_sim,
            'pseudo_loss': pseudo_loss,
            'diversity_loss': diversity_loss,
            'distance_loss': dist_loss
        }
class ClinicalImageBaseClusterDistancePlusGatingModelJ(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelJ, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU())

        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        image_feat, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(image_feat, text_feat)
        outputs = 0.0
        outputs = [self.fusion_fc_list[idx](fuse_features) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        final_output = torch.mean(outputs_tensor, dim=1)
        # outputs = outputs / self.num_experts
        
        return {
            'logits': outputs_tensor,
            'output': final_output,
            'features': fuse_features,
        }
class ClinicalImageBaseClusterDistancePlusGatingModelA0(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelA0, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU()
        )

        self.fusion_fc = nn.Sequential(nn.Linear(192, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        # aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        image_feat, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(image_feat, text_feat)
        outputs = self.fusion_fc(fuse_features)
    
        return {
            'logits': outputs.unsqueeze(1),
            'output': outputs,
            'features': fuse_features,
        }
class ClinicalImageBaseClusterDistancePlusGatingModelJ0(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelJ0, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU())

        self.fusion_fc_list = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
            for _ in range(num_experts)
        ])
        
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        image_feat, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(image_feat, text_feat)
        outputs = 0.0
        outputs = [self.fusion_fc_list[idx](fuse_features) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        final_output = torch.mean(outputs_tensor, dim=1)
        # outputs = outputs / self.num_experts
        
        return {
            'logits': outputs_tensor,
            'output': final_output,
            'features': fuse_features,
        }
class ClinicalImageBaseClusterDistancePlusGatingModelA00(nn.Module):#消融实验的A
    def __init__(self, cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=48, text_dim=25, feat_dim=128, num_experts=3,num_iterations=1,cluster_init_type="kmeans",k=8):
        super(ClinicalImageBaseClusterDistancePlusGatingModelA00, self).__init__()

        # 图像和临床特征提取
        self.encoder = ResNeXt_encoder(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width, widen_factor=widen_factor)
        self.text_encoder = TextNetFeature('none', n_channels=0, num_classes=nlabels, pretrained=False, input_dim=text_dim)
        self.momentum = momentum
        self.alpha = alpha
        self.num_experts = num_experts
        self.feat_dim = feat_dim
        self.stages = self.encoder.stages    
        self.proj = nn.Sequential(
            nn.Linear(self.stages[-1], 128), nn.ReLU(), 
        )

        self.fusion_fc = nn.Sequential(nn.Linear(192, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, nlabels), nn.Sigmoid()) 
        
    def extract_features(self, image, text):#这个是一部分
        # 提取图像特征并融合
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(img_feature_list[-1], img_feature_list[-1].size()[3]).view(img_feature_list[-1].size(0), -1) 
        features_flatten = self.proj(feat_last)

        #对文本特征进行提取
        if text.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            text = text.unsqueeze(1)
        text_feat = self.text_encoder(text)#batch_size, hidden_dim,1 


        return features_flatten, img_feature_list, text_feat#等会做一个没有文本的
    def fuse_features(self, image_features, text_features):
        #DAF模块
        text_features = text_features.view(text_features.size(0), -1)
        assert image_features.size(0) == text_features.size(0), "Batch sizes of image and text features must match."
        
        # 使用torch.cat将两个张量沿着最后一维（特征维度）连接起来
        aligned_features = torch.cat((image_features, text_features), dim=1)
        # aligned_features = self.fuse_proj(aligned_features)
        return aligned_features

    def forward(self, image, text):
        # 提取特征并对齐
        image_feat, features, text_feat = self.extract_features(image, text)#feature_flatten [8, 128]

        fuse_features = self.fuse_features(image_feat, text_feat)
        outputs = self.fusion_fc(fuse_features)
    
        return {
            'logits': outputs.unsqueeze(1),
            'output': outputs,
            'features': fuse_features,
        }
# import torch
# import torch.nn as nn
# import random

# # 假设模型已经被定义为 ClinicalImageALBEFClusterModel
# # batch_size 和输入尺寸的定义
# batch_size = 8
# image_shape = (batch_size, 3, 224, 224)  # 3通道的图片输入
# clinical_data_shape = (batch_size, 25)   # 25维的临床特征输入
# num_iterations = 20  # 定义进行多少次迭代

# # 初始化模型
# model = ClinicalImageBaseClusterDistancePlusGatingModelJ(cardinality=8, depth=29, nlabels=2, alpha=0.5, momentum =0.998, base_width=64, widen_factor=4, queue_size=4, feat_dim=128, num_experts=3,num_iterations=20,cluster_init_type="kmeans++")

# # 将模型设为训练模式
# model.train()

# # 测试循环，进行多次迭代
# for iteration in range(num_iterations):
#     print(f"--- Iteration {iteration + 1} ---")
    
#     # 生成随机的图像输入和临床特征输入
#     image_input = torch.randn(image_shape)  # 随机生成 batch_size 个 3x224x224 的图片
#     clinical_input = torch.randn(clinical_data_shape)  # 随机生成 batch_size 个 25 维的临床特征
    
#     # 将数据传入模型并执行前向传播
#     outputs = model(image_input, clinical_input)
    
#     # 输出每个部分的形状，如果对应的键存在
#     if 'logits' in outputs:
#         print("Logits shape:", outputs['logits'].shape)
#     if 'output' in outputs:
#         print("Final output shape:", outputs['output'].shape)
#     if 'features' in outputs:
#         print("Extracted features shape:", outputs['features'].shape)
#     if 'pseudo_labels' in outputs:
#         print("Pseudo labels shape:", outputs['pseudo_labels'].shape)
#     # 如果diversity_loss存在，则输出它
#     if 'diversity_loss' in outputs:
#         print("Diversity loss:", outputs['diversity_loss'])  # 这是一个标量
#     if 'sim_loss' in outputs:
#         print("Similarity loss:", outputs['sim_loss'])  # 这是一个标量
#     if 'distance_loss' in outputs:
#         print("Distance loss:", outputs['distance_loss'])  # 这是一个标量
#     print("\n")

# print("Model testing completed.")