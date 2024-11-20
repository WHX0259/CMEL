'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
import random
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
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
    def __init__(self, cardinality=8, depth=29, base_width=64, widen_factor=4):
        super(ResNeXt_encoder, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
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
        feat_last = F.avg_pool2d(feat_list[-1], feat_list[-1].size()[3]).view(feat_list[-1].size(0), -1) 

        return feat_list, feat_last#获得特征列表

class CrossModalAttention(nn.Module):#对齐再融合
    def __init__(self, img_dim, clinical_dim, output_dim):
        super(CrossModalAttention, self).__init__()
        self.clinical_query = nn.Linear(clinical_dim, output_dim)  # 图像特征的 Query
        self.img_key = nn.Linear(img_dim, output_dim)  # 临床特征的 Key
        self.img_value = nn.Linear(img_dim, output_dim)  # 临床特征的 Value
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img_features, clinical_features):
        query = self.clinical_query(clinical_features)
        key = self.img_key(img_features)#8，128
        value = self.img_value(img_features)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = self.softmax(attention_scores)
        aligned_features = torch.matmul(attention_weights, value)
        
        output = query + aligned_features
        return output#进行对齐了
#加油我可以把它改好
class GCALBEF(nn.Module):
    def __init__(self,          
                 img_encoder = ResNeXt_encoder,       
                 text_encoder = TextNetFeature,
                 clinic_length = 22,   
                 temp = 0.07,
                 queue_size = 128,
                 momentum = 0.998,
                 feat_dim=128,
                 num_classes = 2,
                 ):
        super().__init__()
        embed_dim = feat_dim
     
        self.visual_encoder = img_encoder()
        vision_width = self.visual_encoder.stages[-1]    
        # bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = text_encoder(input_dim=clinic_length,n_channels=[128],backbone=None,pretrained=None,num_classes=None) 

        text_width = self.text_encoder.hidden[-1]
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * temp)   
        self.queue_size = queue_size
        self.momentum = momentum
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        self.visual_encoder_m = img_encoder() 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = text_encoder(input_dim=clinic_length,n_channels=[128],backbone=None,pretrained=None,num_classes=None)     
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        self.cross_attention = CrossModalAttention(embed_dim, embed_dim, feat_dim)
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(feat_dim, num_classes), nn.Sigmoid()) 


    def forward(self, image, clinic, alpha=0, mode = 'train'):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        if clinic.dim() == 2:  # 如果张量是二维的
            # 在第二个位置插入一个新的维度
            clinic = clinic.unsqueeze(1)
        _,image_features = self.visual_encoder(image) 
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_features),dim=-1)  

        text_output = self.text_encoder(clinic)            
        text_feat = F.normalize(self.text_proj(text_output.view(text_output.size(0), -1)),dim=-1)                 
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            _, image_features_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_features_m),dim=-1)  #如果是ViT应该只是使用了cls token
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            text_output_m = self.text_encoder_m(clinic)   
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.view(text_output.size(0),-1)),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        if mode == 'train':
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)#image_feat_m 4, 128  4, 128
        aligned_feat = self.cross_attention(image_feat,text_feat)
        output = self.fusion_fc(aligned_feat)

        return {
            'ita_loss':loss_ita,
            'output': output,
            'feature': aligned_feat,
            'logits': output.unsqueeze(1),
        }  

        

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
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


## 配置参数
# config = {'embed_dim': 128}  # 假设嵌入维度为512
# img_encoder = ResNeXt_encoder
# text_encoder = TextNetFeature
# clinic_length = 22

# # 初始化模型
# model = GCALBEF(img_encoder=img_encoder, text_encoder=text_encoder, clinic_length=clinic_length, config=config)

# # 生成随机数据
# batch_size = 4
# image = torch.randn(batch_size, 3, 224, 224)  # 随机生成一批图像
# clinic = torch.randn(batch_size, clinic_length)  # 随机生成一批文本数据

# # 设置模型为评估模式
# model.eval()

# # 不需要计算梯度
# with torch.no_grad():
#     # 前向传播
#     output = model(image, clinic)

# # 打印输出
# print("ITA Loss:", output['ita_loss'].item())
# print("Output:", output['output'].shape)
# print("Feature:", output['feature'].shape)