import argparse
import math
import os
import sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score,precision_recall_curve
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import csv
from torch.cuda.amp import autocast
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import pickle
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import OrderedDict
os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '8854'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from lib.utils.config_ import get_raw_dict
from lib.dataset.get_text_mask_manual import Dataset_Slice_text_manual#数据读取的
from lib.utils.logger import setup_logger
from lib.utils.loss import build_loss, Diversity_loss
from lib.utils.metrics import analysis_pred_binary
#ClinicalImageBaseClusterDistancePlusGatingModel是CMEL
from lib.model.CMEL import ClinicalImageBaseClusterDistancePlusGatingModel, ClinicalImageALBEFClusterDistancePlusGatingModel
from lib.model.ablationExperiment import ClinicalImageBaseClusterDistancePlusGatingModelH, ClinicalImageBaseClusterDistancePlusGatingModelJ, ClinicalImageBaseClusterDistancePlusGatingModelG,ClinicalImageBaseClusterDistancePlusGatingModelF,ClinicalImageBaseClusterDistancePlusGatingModelE, ClinicalImageBaseClusterDistancePlusGatingModelD, ClinicalImageBaseClusterDistancePlusGatingModelC, ClinicalImageBaseClusterDistancePlusGatingModelB, ClinicalImageBaseClusterDistancePlusGatingModelA0,ClinicalImageBaseClusterDistancePlusGatingModelJ0, ClinicalImageBaseClusterDistancePlusGatingModelA,ClinicalImageBaseClusterDistancePlusGatingModelA00
from lib.model.PathomicFusion import PathomicNet, PathgraphomicNet
from lib.model.samms import SammsNet
from lib.model.ALBEF import GCALBEF
# default_collate_func = torch.utils.data.dataloader.default_collate

#下面是参数定义
def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    #下面是参数设置
    #主要调节下面这些变量
    parser.add_argument('--fold', type=int,default=3,
                        help="Name of the fold to use")
    parser.add_argument('--slice_path', type=str,default=r'/data/wuhuixuan/data/padding_crop',#存储数据的
                        help="Name of the fold to use")
    parser.add_argument('--fold_json', type=str,default=r'/data/huixuan/data/data_chi/TRG_patient_folds.json',
                        help="Name of the fold to use")
    parser.add_argument('--manual_csv_path', type=str,default=r'/data/wuhuixuan/code/Self_Distill_MoE/data/selected_features_22_with_id_label_fold_norm.csv',
                        help="Name of the fold to use")
    parser.add_argument('--sentence_json', type=str,default=r'/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/sentences.json',
                        help="Name of the fold to use")
    parser.add_argument('--csv_path', type=str,default=r'/data/huixuan/data/data_chi/label.csv',
                        help="Name of the fold to use")
    parser.add_argument('--display_experts', type=bool,default=False,#这个是要不要展示所有专家的结果
                        help="Display every experts result or not.")
    parser.add_argument('--input_radiomic', type=bool,default=False,#这个是要不要展示所有专家的结果
                        help="Display every experts result or not.")
    parser.add_argument('--criterion_type', type=str,default='ajs_uskd_mixed',
                        help="Name of the criterion to use")
    parser.add_argument('--diversity_metric', type=str,default='erm',#erm是没有多样性损失，var是方差计算的多样性损失
                        help="Name of the criterion to use")
    parser.add_argument('--lambda_v', type=float,default=0.5,#这个是多样性损失前面的系数，
                        help="Name of the criterion to use")
    parser.add_argument('--text_dim', type=int,default=25,
                        help="The dim of the input text")
    #下面这个是结果保存的位置，我保存为了csv文件
    parser.add_argument('--excel_file_name', help='note', default=None)
    parser.add_argument('--note', help='note', default='Causal experiment')
    parser.add_argument('--model_type',type=str, default = None,help='note')
    #maybe you should change the output path
    parser.add_argument('--output', default='/data16t/huixuan/code/Multi_Modal_MoE/lib/output')#输出
    parser.add_argument('--num_class', default=2, type=int,
                        help="Number of query slots")#interactionValue
    parser.add_argument('--interactionValue', default=0.2, type=float,#这个咱这个模型用不到，就是这个是我设计的双分支的时候用到的两个分支之间交互的比例
                        help="Interaction Value of Interaction modules")#interactionValue
    parser.add_argument('--num_experts', default=3, type=int,#专家个数或者分类头的个数
                        help="Number of experts")
    parser.add_argument('--logit_method', default='ClinicalImageBaseClusterDistancePlusGatingModel', type=str,
                        help="Number of query slots")#ClinicalImageBaseClusterDistancePlusGatingModel这个模型就是使用了临床信息和图像信息进行聚类的有距离损失和使用Kmeans的模型
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',#模型预训练使用预训练的参数
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],#优化器
                        help='which optim to use')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    # # loss loss_final_value
    # parser.add_argument('--use_criterion_total', default=True, type=bool,
    #                     help="Use or don't use total criterion")
    parser.add_argument('--eps', default=1e-5, type=float,#防止为0，暂时没找到在哪用的
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--gamma', default=2.0, type=float,
                        metavar='gamma', help='gamma for focal loss')
    parser.add_argument('--alpha', default=0.25, type=float,
                        metavar='alpha', help='alpha for focal loss')#mixed loss中的focal用的
    parser.add_argument('--focal_weight', type=float, default = 1.0, help='List of Loss weight')
    parser.add_argument('--sim_weight', default= 0.0001, type=float,#这个是这两个损失的权重
                        metavar='alpha', help='alpha for focal loss')
    parser.add_argument('--dis_weight', default= 0.0001, type=float,
                        metavar='alpha', help='alpha for focal loss')
    parser.add_argument('--loss_final_value', default=0.0, type=float,#这个是pesudo最后占比
                        metavar='loss_final_value', help='logit loss and output loss')
    parser.add_argument('--loss_dev', default=2, type=float,
                        help='scale factor for loss')#不知道，是把一些值变成负的
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',#epoch
                        help='number of total epochs to run')
    parser.add_argument('--val_interval', default=1, type=int, metavar='N',#原来是1哈，就是多少个epoch之后进行验证
                        help='interval of validation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',#是不是从头开始训练
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,#batch_size
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,#学习率我感觉可以低
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,#
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,#结果就是这个没用到每一轮都得print
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume',  type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model_path',  type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',#是不是测试模式
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',#这个是ema model训练的参数我们在自己模型中定义了这个我们就去掉
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

    # distribution training，下面是分布式训练的东西，还没涉及到
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=1024, type=int,#extra_dim
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--extra_dim', default=22, type=int,#extra_dim,临床信息的话维度是25，机器学习特征的话维度是22
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.2, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining，是不是提前停止等，在参数没有太大变化或者在损失爆炸的时候停止
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args
best_mAP = 0
best_meanAUC = 0
best_f1_score = 0

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def main():
    args = get_args()
    #调试添加参数
    args.cls_num_list = [150, 224]

    args.excel_file_name = f'/data16t/huixuan/code/Multi_Modal_MoE/lib/result/MultiExperts_{args.model_type}_{args.num_experts}_{args.criterion_type}_{args.diversity_metric}_Interaction_{args.interactionValue}__sim_{args.sim_weight}_dis_{args.dis_weight}_lfv_{args.loss_final_value}_Adjust_threshold.csv'
        
    args.output = ensure_directory_exists(os.path.join(args.output, args.model_type))
    args.output = ensure_directory_exists(os.path.join(args.output, f'num_expert{args.num_experts}'))
    args.output = ensure_directory_exists(os.path.join(args.output, args.criterion_type))
    args.output = ensure_directory_exists(os.path.join(args.output, str(args.fold + 1)))
    args.output = ensure_directory_exists(os.path.join(args.output, args.diversity_metric))
    args.output = ensure_directory_exists(os.path.join(args.output, f"Interaction_{args.interactionValue}"))
    args.output = ensure_directory_exists(os.path.join(args.output, f"lfv_{args.loss_final_value}"))
    args.output = ensure_directory_exists(os.path.join(args.output, f"sim_{args.sim_weight}"))
    args.output = ensure_directory_exists(os.path.join(args.output, f"dis_{args.dis_weight}"))
    print(args.output)
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="CAUSAL")
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP
    global best_meanAUC
    global best_f1_score
    # args.resume=f"/data/wuhuixuan/code/Self_Distill_MoE/out/{args.model_type}/{args.criterion_type}/{args.fold+1}/train/{args.logit_method}/model_best.pth.tar"
    # Build model
    if args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModel":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModel(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=20,cluster_init_type="kmeans++",k=8).cuda()
    #ClinicalImageALBEFClusterDistancePlusGatingModel
    elif args.model_type == "ClinicalImageALBEFClusterDistancePlusGatingModel":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageALBEFClusterDistancePlusGatingModel(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++").cuda()
    #ImageBaseClusterDistancePlusGatingModel
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelE":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelE(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelD":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelD(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelC":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelC(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    #ClinicalImageBaseClusterDistancePlusGatingModelB
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelB":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelB(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelA":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelA(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelF":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelF(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelG":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelG(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelH":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelH(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelJ":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelJ(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelJ0":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelJ0(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "PathomicNet":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = PathomicNet( nlabels=args.num_class).cuda()
    elif args.model_type == "PathgraphomicNet":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = PathgraphomicNet( nlabels=args.num_class).cuda()#SammsNet
    elif args.model_type == "SammsNet":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = SammsNet( nlabels=args.num_class).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelA0":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelA0(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == "ClinicalImageBaseClusterDistancePlusGatingModelA00":#, num_experts=3,num_iterations=20,cluster_init_type="kmeans++"
        model = ClinicalImageBaseClusterDistancePlusGatingModelA00(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == 'GCALBEF':#ClinicalImageBranchClusterModel
        model = GCALBEF(clinic_length = args.text_dim, num_classes = args.num_class).cuda()
    model = model.cuda()#ClinicalImageALBEFClusterModel
    model = torch.nn.DataParallel(model)
    criterion = build_loss(args.criterion_type, args)

    # Optimizer
    args.lr_mult = args.batch_size / 256#根据batch_size大小调整学习率
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # Tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    if args.resume:
        if os.path.isfile(args.resume):#读取模型参数
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                state_dict = checkpoint
            logger.info("Omitting {}".format(args.resume_omit))
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    best_result = []
    best_meanAUC = 0

    train_dataset = Dataset_Slice_text_manual(
        slice_path=args.slice_path,
        fold_json=args.fold_json,
        manual_csv_path=args.manual_csv_path,
        sentence_json=args.sentence_json,
        csv_path=args.csv_path,
        fold=args.fold, mode='train'
    )
    val_dataset = Dataset_Slice_text_manual(
        slice_path=args.slice_path,
        fold_json=args.fold_json,
        manual_csv_path=args.manual_csv_path,
        sentence_json=args.sentence_json,
        csv_path=args.csv_path,
        fold=args.fold, mode='val'
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, drop_last=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, sampler=val_sampler, drop_last=True
    )

    if args.evaluate:
        args.output = os.path.join(args.output, 'val')#测试得到的结果放到这个文件中
        os.makedirs(args.output, exist_ok=True)
        args.output = os.path.join(args.output, args.logit_method)
        os.makedirs(args.output, exist_ok=True)
        metrics, avg_metrics, net_benefit = test(train_loader, val_loader, model, args, logger, thresholds=0.5)
        logger.info(' * Average Metrics: {}'.format(avg_metrics))
        return
    # Criterion难道十折
    args.output = os.path.join(args.output, 'train')
    #  args.output = os.path.join(args.output, args.model_type)
    if not os.path.exists(args.output):  # 加入模型类型
        os.makedirs(args.output)
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses],
        prefix='=> Test Epoch: '
    )

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_epoch = -1
    train_loss_list = []
    val_loss_list1 = []
    val_loss_list2 = []
    torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()

        startt = time.time()
        loss, train_metrics = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)
        train_loss_list.append(loss)
        endt = time.time()
        logger.info("Time used：    {} seconds".format(endt - startt))

        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch % args.val_interval == 0:
            val_loss, val_metrics = validate(val_loader, model, criterion, args, logger, epoch)
            val_loss_list1.append(val_loss)
            val_loss_ema, val_metrics_ema = validate(val_loader, ema_m.module, criterion, args, logger, epoch)
            val_loss_list2.append(val_loss_ema)

            losses.update(val_loss)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))
            progress.display(epoch, logger)

            save_path = os.path.join(args.output, args.model_type)
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, 'checkpoint.pth.tar')

            is_best = False
            if val_metrics['AUC'] > val_metrics_ema['AUC']:
                if val_metrics['AUC'] > best_meanAUC:
                    best_meanAUC = val_metrics['AUC']
                    best_epoch = epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_meanAUC': best_meanAUC,
                        'optimizer': optimizer.state_dict(),
                    }, is_best=True, filename=filename)
                    logger.info("{} | Set best meanAUC {} in ep {}".format(epoch, best_meanAUC, best_epoch))
            else:
                if val_metrics_ema['AUC'] > best_meanAUC:
                    best_meanAUC = val_metrics_ema['AUC']
                    best_epoch = epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': ema_m.module.state_dict(),
                        'best_meanAUC': best_meanAUC,
                        'optimizer': optimizer.state_dict(),
                    }, is_best=True, filename=filename)
                    logger.info("{} | Set best meanAUC {} in ep {}".format(epoch, best_meanAUC, best_epoch))

            if not is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_meanAUC': best_meanAUC,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=filename)

            if math.isnan(val_loss) :
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_meanAUC': best_meanAUC,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            if args.early_stop and epoch - best_epoch > 8:
                logger.info("Early stopping at epoch {} with best epoch at {}".format(epoch, best_epoch))
                break

    print("Best mAP:", best_mAP)
    best_result.append(best_mAP)
    plot_losses(train_loss_list, val_loss_list1, val_loss_list2, args.output)
    if summary_writer:
        summary_writer.close()

    best_mAP = sum(best_result) / len(best_result)
    print("Best mAP:", best_mAP)
    return 0


#把每个output都变成list的处理方式, 我需要保存每个专家的结果和评价指标
def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))#调整学习率
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    ema_m.eval()
    end = time.time()

    all_targets = []
    all_logits = []
    all_outputs = []  # Assuming three heads
    for i, (images, clinic_feature, manual_features, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
        clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.model_type in ["ClinicalImageBranchClusterModel","ClinicalImageBranchClusterModel_Lv2","ClinicalImageBranchClusterModel_Lv3","ClinicalImageBranchClusterGatingModel"]:
                outputs = model(images, clinic_feature,epoch)
            elif args.input_radiomic:
                outputs = model(images, clinic_feature,manual_features)
            else:
                outputs = model(images, clinic_feature)
            loss = 0.0
            output = outputs['output']
            logits = outputs['logits']
            # if args.criterion_type == 'AJS_USKD':
            #     features = outputs['features']
            logits_list = list(torch.unbind(logits, dim=1))
            prob_y = torch.sigmoid(logits)
            one_hot_target = F.one_hot(target, num_classes=args.num_class)
            prob_y_stack = torch.sigmoid(logits)
            probs_y = []
            for i in range(args.num_experts):#这个计算多次分类头输出用这个计算的多个输出进行后续计算，多样性损失
                prob_y = prob_y_stack[:,i,:]
                if args.diversity_metric == 'var':
                    prob_y_vec = torch.masked_select(input=prob_y, mask=one_hot_target.bool())#计算目标类别预测的多样性，利用方差进行衡量          
                    probs_y.append(prob_y_vec.unsqueeze(0))#这里就计算
                else:
                    loss_diversity = 0.0
            loss_extra=0.0
            alpha = sigmoid_increase(epoch, args.epochs, final_value=args.loss_final_value)
            if 'sim_loss' in outputs:#'dist_loss' in outputs and 
                loss_extra = outputs['sim_loss'] * args.sim_weight#outputs['dist_loss'] * args.dis_weight+  
                if 'pseudo_loss' in outputs:
                    loss_extra += alpha*outputs['pseudo_loss']
                    print('pseudo loss: {:.3f} alpha: {:.3f}'.format(outputs['pseudo_loss'], alpha))
                if 'diversity_loss' in outputs:
                    loss_extra += outputs['diversity_loss']
                    print('diversity loss: {:.3f}'.format(outputs['diversity_loss']))
                if 'distance_loss' in outputs:
                    loss_extra += args.dis_weight *outputs['distance_loss']
                    print('distance loss: {:.3f} '.format(outputs['distance_loss']))
            elif 'ita_loss' in outputs:#'dist_loss' in outputs and 
                loss_extra = 0.1*outputs['ita_loss']
            # loss = alpha*criterion(output, target)+(1-alpha)*criterion(logits_list, target)# output_logits, targets, extra_info=None, return_expert_losses=False# output_logits, targets, extra_info=None, return_expert_losses=False
            image_out = outputs.get('image_out', None)
            clinic_out = outputs.get('clinic_out', None)
            if (image_out is not None) and (clinic_out is not None):
                loss = criterion(clinic_out, target)+alpha*criterion(image_out, target)+(1-alpha)*criterion(logits_list, target)
            else:#加了变化的损失没有什么用还降低了性能
                loss = alpha*criterion(output, target)+(1-alpha)*criterion(logits_list, target)# output_logits, targets, extra_info=None, return_expert_losses=Fals
            
            if args.diversity_metric == 'erm':
                loss_diversity = 0.0
            elif args.diversity_metric == 'var':
                probs_y = torch.cat(probs_y, dim=0)
                X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log(2*probs_y/(1+probs_y)) + 1e-6)
                loss_diversity = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
            elif args.diversity_metric == 'difference':
                loss_diversity = Diversity_loss(logits, target)
            elif args.diversity_metric == 'js':
                loss_diversity = -average_js_divergence_among_experts(logits)
            else:
                raise NotImplementedError
            loss = loss + loss_extra- args.lambda_v * loss_diversity#loss4.2, loss_diversity 0.0014
            logger.info('loss: {:.3f}'.format(loss))

            if args.loss_dev > 0:
                loss *= args.loss_dev

        losses.update(loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        all_logits.append(logits.detach().cpu().numpy())
        all_outputs.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        lr.update(get_learning_rate(optimizer))

        if epoch > args.ema_epoch:
            ema_m.update(model)
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_logits = np.concatenate(all_logits)

    metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_logits, all_outputs)
    log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='train', logger=logger,epoch=epoch, loss = losses.avg)#args=args, metrics=metrics, avg_metrics=avg_metrics, mode='test', logger=logger

    return losses.avg, avg_metrics

def validate(val_loader, model, criterion, args, logger, epoch):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    model.eval()
    save_path = os.path.join(args.output, 'validation_results.csv')

    all_targets = []
    all_logits = []
    all_outputs = []
    # extra_info = {}
    with torch.no_grad():
        end = time.time()
        for i, (images, clinic_feature, manual_features, target, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.model_type in ["ClinicalImageBranchClusterModel","ClinicalImageBranchClusterModel_Lv2","ClinicalImageBranchClusterModel_Lv3","ClinicalImageBranchClusterGatingModel"]:
                    outputs = model(images, clinic_feature,epoch)
                elif args.input_radiomic:
                    outputs = model(images, clinic_feature,manual_features)
                else:
                    outputs = model(images, clinic_feature)
                output = outputs['output']
                logits = outputs['logits']
                prob_y_stack = torch.sigmoid(logits)
                probs_y = []
                one_hot_target = F.one_hot(target, num_classes=args.num_class)
                for i in range(args.num_experts):#这个计算多次分类头输出用这个计算的多个输出进行后续计算
                    prob_y = prob_y_stack[:,i,:]
                    if args.diversity_metric == 'var':
                        prob_y_vec = torch.masked_select(input=prob_y, mask=one_hot_target.bool())          
                        probs_y.append(prob_y_vec.unsqueeze(0))
                    else:
                        loss_diversity = 0.0
                # features = outputs['features']
                logits_list = list(torch.unbind(logits, dim=1))
                loss_extra=0.0
                alpha = sigmoid_increase(epoch, args.epochs, final_value=args.loss_final_value)
                if 'sim_loss' in outputs:#'dist_loss' in outputs and 
                    loss_extra = args.sim_weight * outputs['sim_loss']#outputs['dist_loss'] * args.dis_weight + 
                    if 'pseudo_loss' in outputs:
                        loss_extra += alpha*outputs['pseudo_loss']
                    if 'diversity_loss' in outputs:
                        loss_extra += outputs['diversity_loss']
                    if 'distance_loss' in outputs:
                        loss_extra += args.dis_weight *outputs['distance_loss']
                if args.diversity_metric == 'erm':
                    loss_diversity = 0.0
                elif args.diversity_metric == 'var':
                    probs_y = torch.cat(probs_y, dim=0)
                    X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log((2*probs_y + 1e-6)/(1+probs_y)) + 1e-6)
                    loss_diversity = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
                elif args.diversity_metric == 'difference':
                    loss_diversity = Diversity_loss(logits, target)
                elif args.diversity_metric == 'js':
                    loss_diversity = -average_js_divergence_among_experts(logits)
                else:
                    raise NotImplementedError
                # loss = criterion(logits_list, target)# output_logits, targets, extra_info=None, return_expert_losses=False
                loss = alpha*criterion(output, target)- args.lambda_v * loss_diversity + (1-alpha)*criterion(logits_list, target)
                # similarity_scores = calculate_similarity_scores(prob_y_stack, target)#每个样本都有三个值
                image_out = outputs.get('image_out', None)
                clinic_out = outputs.get('clinic_out', None)
                if (image_out is not None) and (clinic_out is not None):
                    loss = criterion(clinic_out, target)+alpha*criterion(image_out, target)+(1-alpha)*criterion(logits_list, target)
                else:
                    loss = alpha*criterion(output, target)+(1-alpha)*criterion(logits_list, target)# output_logits, targets, extra_info=None, return_expert_losses=Fals

                loss = loss + loss_extra
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                all_outputs.append(output.detach().cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())

            losses.update(loss.item() * args.batch_size, images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        if dist.get_world_size() > 1:
            dist.barrier()

        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        all_outputs = np.concatenate(all_outputs)
        # Apply softmax to logits for saving probabilities
        all_probabilities = F.softmax(torch.tensor(all_logits), dim=2).numpy()

        # Calculate metrics
        metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_logits,all_outputs)
        log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='val', logger=logger, epoch=epoch, loss=losses.avg)
        return losses.avg, avg_metrics



def test(train_loader, val_loader, model, args, logger, thresholds):
    #怎么说呢，我之后聚类的对象就不是三个专家的特征了而是单个特征，所以需要一个判断语句或者一个指示的参数指明我需要怎么画T-SNE图
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    # best_threshold = 0.5
    # best_thresholds = [0.5, 0.5]
    saved_data = []
    all_features = []
    all_targets = []
    all_outputs = []  
    net_benefit = []
    # all_pesudo_label = []
    all_image_output = []
    all_clinic_output = []
    model_name_xgb = f'/data/wuhuixuan/code/Causal-main/save/XGBoost/xgb_model_fold_{args.fold + 1}.pkl'
    with open(model_name_xgb, 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    model_name_lgbm = f'/data/wuhuixuan/code/Causal-main/save/lgbm/lgbm_model_fold_{args.fold + 1}.pkl'
    with open(model_name_lgbm, 'rb') as model_file:
        lgbm_model = pickle.load(model_file)
    
    # if args.logit_method == args.model_type:
    train_features, train_labels, train_predictions, train_pseudo_labels = extract_features(train_loader, model, args)#等会别忘改
    val_features, val_labels, _, val_pseudo_labels = extract_features(val_loader, model, args)
    selected_features_data = pd.read_csv('/data/wuhuixuan/code/Causal-main/data/selected_features_22_with_id_label_fold.csv')
    id_to_index = {str(row['ID']): idx for idx, row in selected_features_data.iterrows()}
    final_outputs = []
    with torch.no_grad():
        for i, (images, clinic_feature, manual_features, target, idx) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            with autocast(enabled=args.amp):
                if args.model_type in ["ClinicalImageBranchClusterModel","ClinicalImageBranchClusterModel_Lv2","ClinicalImageBranchClusterModel_Lv3","ClinicalImageBranchClusterGatingModel"]:
                    outputs = model(images, clinic_feature,epoch=args.epochs-1)
                elif args.input_radiomic:
                    outputs = model(images, clinic_feature,manual_features)
                elif args.model_type in ["GCALBEF"]:
                    outputs = model(images, clinic_feature,mode = 'val')
                else:
                    outputs = model(images, clinic_feature)
                output = outputs['output']
                logits = outputs['logits']
                pseudo_labels = outputs.get('pseudo_labels', None)
                image_out = outputs.get('image_out', None)
                clinic_out = outputs.get('clinic_out', None)
                # 检查并提取 'image_out' 和 'clinic_out' 的值
                    
            # features = outputs['features'].detach().cpu().numpy()
            output_cpu = output.detach().cpu().numpy()
            # all_features.append(features)
            all_outputs.append(logits.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            if image_out is not None:
                all_image_output.append(image_out.cpu().numpy())
            if clinic_out is not None:
                all_clinic_output.append(clinic_out.cpu().numpy())
            # all_pesudo_label.append(pseudo_labels.cpu().numpy())
            batch_features = []
            mapped_labels = []
            for idx_val in idx.numpy():
                feature_idx = id_to_index.get(str(idx_val) + '.nii.gz')
                if feature_idx is not None:
                    mapped_labels.append(selected_features_data.iloc[feature_idx]['label'])
                    batch_features.append(selected_features_data.drop(columns=['ID', 'label', 'fold']).iloc[feature_idx].values)
                else:
                    mapped_labels.append(None)
                    batch_features.append(np.zeros(selected_features_data.drop(columns=['ID', 'label', 'fold']).shape[1]))

            batch_features = np.array(batch_features)

            xgb_predictions = xgb_model.predict_proba(batch_features)
            lgbm_predictions = lgbm_model.predict_proba(batch_features)

            final_output = F.softmax(torch.tensor(output_cpu), dim=1).numpy()
            final_outputs.append(final_output)
            # Get softmax outputs for each expert and final output
            expert_outputs = [F.softmax(logits[:, i, :].detach().cpu(), dim=1).numpy()[:, 1] for i in range(logits.shape[1])]
            final_output_class1 = final_output[:, 1]  # Get the probability of class 1 for final output
           
            assert all(len(expert_outputs[i]) == logits.shape[0] for i in range(len(expert_outputs))), "Lengths must match."
            # 构建 CSV 数据
            for i in range(logits.shape[0]):
                index = idx[i].item()
                target_val = target[i].item()
                row = [index, target_val]
                row.extend(expert_outputs[j][i] for j in range(len(expert_outputs)))
                row.append(final_output_class1[i])
                if pseudo_labels is not None:
                    row.append(pseudo_labels[i])
                if image_out is not None:
                    row.append(image_out[i])
                if clinic_out is not None:
                    row.append(clinic_out[i])
                    
                saved_data.append(row)
            # Calculate net benefit for Decision Curve Analysis
            X_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}'].drop(
                columns=['ID', 'label', 'fold'])
            y_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}']['label']
            y_test_prob = xgb_model.predict_proba(X_test)[:, 1]

            net_benefit.append(calculate_net_benefit(y_test, y_test_prob, thresholds))

    # Convert saved_data to DataFrame and save to CSV
    # saved_data = np.array(saved_data)
    save_path = os.path.join(args.output, 'save_data.csv')
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        header = ['ID', 'Target']
        header.extend([f'Expert_{i}' for i in range(len(expert_outputs))])
        header.append('Avg_Output')
        writer.writerow(header)
        
        # 写入数据
        for row in saved_data:
            writer.writerow(row)
    # df = pd.DataFrame(saved_data, columns=columns)
    # df.to_csv(save_path, index=False)
    # print(columns)
    print(save_path)
    # df.to_csv(save_path, index=False)

    # Concatenate all final_outputs to form a single array
    all_final_outputs = np.concatenate(final_outputs, axis=0)
    # Calculate performance metrics
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    # all_features = np.vstack(all_features)
    
    if all_image_output and all_clinic_output:
        all_image_output = np.concatenate(all_image_output)
        all_clinic_output = np.concatenate(all_clinic_output)
        image_metrics = calculate_binary_metrics(all_targets, all_image_output)#args, metrics, mode, epoch, output_class,loss=None, logger=None, threshold=0.5
        clinic_metrics = calculate_binary_metrics(all_targets, all_clinic_output)
        #args, metrics, mode, epoch, output_class,loss=None, logger=None, threshold=0.5
        log_and_save_binary_metrics(args,image_metrics,'test',100,"image_output",loss = None, logger=logger, threshold =0.5)
        log_and_save_binary_metrics(args,clinic_metrics,'test',100,"clinic_output",loss = None, logger=logger, threshold =0.5)
    # all_pesudo_label = np.concatenate(all_pesudo_label)
    all_target_onehot = one_hot(all_targets)
    metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_outputs,all_final_outputs)
    substring = "Base"

    #下面全都注释了
    if args.display_experts and args.logit_method == args.model_type:
        if substring in args.model_type:
            tsne_base_plot(train_features, train_labels, train_pseudo_labels, val_features, val_labels, val_pseudo_labels,filename=os.path.join(args.output,'tsne_plot_all_experts.png'))
            pca_base_plot(train_features, train_labels, train_pseudo_labels, val_features, val_labels, val_pseudo_labels,filename=os.path.join(args.output,'pca_plot_all_experts.png'))
        else:
            plot_all_experts_tsne(train_features, train_labels, val_features, val_labels, num_classes=2, experts_range=range(args.num_experts), filename=os.path.join(args.output,'tsne_plot_all_experts.png'))
            plot_all_experts_pca(train_features, train_labels, val_features, val_labels, num_classes=2, experts_range=range(args.num_experts), filename=os.path.join(args.output,'pca_plot_all_experts.png'))
            # plot_all_experts_select_feat_tsne(train_features, train_labels, val_features, val_labels, train_pseudo_labels =train_pseudo_labels, val_pseudo_labels =val_pseudo_labels, num_classes=2, filename=os.path.join(args.output,'tsne_select_plot.png'))
            # plot_all_experts_select_feat_pca(train_features, train_labels, val_features, val_labels, train_pseudo_labels =train_pseudo_labels, val_pseudo_labels =val_pseudo_labels, num_classes=2, filename=os.path.join(args.output,'pca_select_plot.png'))

            softmax_outputs = [F.softmax(torch.tensor(all_outputs[:, expert_idx, :]), dim=1).numpy() for expert_idx in range(args.num_experts)]
            for expert_idx in range(args.num_experts):
                # expert_features = all_features[:, expert_idx, :]#features, labels, num_classes, filename='tsne_plot.png'
                plot_tsne(train_features[: ,expert_idx ,: ], train_labels, val_features[: ,expert_idx ,: ], val_labels, num_classes=2, filename=os.path.join(args.output,f'tsne_plot_expert_{expert_idx}.png'))
                plot_pca(train_features[: ,expert_idx ,: ], train_labels, val_features[: ,expert_idx ,: ], val_labels, num_classes=2, filename=os.path.join(args.output,f'pca_plot_expert_{expert_idx}.png'))
                plot_roc_curve(all_target_onehot, softmax_outputs[expert_idx], num_classes=2, filename=os.path.join(args.output,f'roc_curve_expert_{expert_idx}.png'))
                plot_dca_curves(all_targets, softmax_outputs[expert_idx], filename=os.path.join(args.output,f'dca_curves_expert_{expert_idx}.png'))
    else:
        metrics = None
            # avg_output = np.mean(all_outputs, axis=1)
    avg_softmax_output = all_final_outputs
    plot_roc_curve(all_target_onehot, avg_softmax_output, num_classes=2, filename=os.path.join(args.output,'roc_curve_avg_experts.png'))
    plot_pr_curve(all_target_onehot, avg_softmax_output, num_classes=2, filename=os.path.join(args.output,'pr_curve_avg_experts.png'))
    plot_dca_curves(all_targets, avg_softmax_output, filename=os.path.join(args.output,f'dca_curves_expert_avg.png'))
        # plot_tsne(all_features_mean, all_targets, num_classes=2)
    log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='test', logger=logger, epoch=args.epochs,threshold =0.5)

    return metrics, avg_metrics, net_benefit

############################################下面就是定义的函数#########################################

def plot_pca(train_features, train_labels, val_features, val_labels, num_classes, filename='pca_plot.png'):
    pca = PCA(n_components=2)
    all_features = np.concatenate([train_features, val_features])
    
    # Check if all_features is valid
    if all_features.shape[1] == 0:
        raise ValueError("The input features for PCA are empty.")
    
    pca_results = pca.fit_transform(all_features)
    
    train_pca_results = pca_results[:len(train_features)]
    val_pca_results = pca_results[len(train_features):]
    
    # Define the colors: pink and blue
    colors = ['#FF69B4', '#1E90FF']  # Pink and blue

    plt.figure(figsize=(10, 10))
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        plt.scatter(train_pca_results[train_indices, 0], train_pca_results[train_indices, 1], 
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx % len(colors)])
        
        plt.scatter(val_pca_results[val_indices, 0], val_pca_results[val_indices, 1], 
                    label=f'Val Class {class_idx}', alpha=0.8, edgecolor='k', color=colors[class_idx % len(colors)])
    
    plt.legend()
    plt.title("PCA of Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_select_pca(train_features, train_labels, train_pseudo_labels, val_features, val_labels, val_pseudo_labels, num_classes, filename='pca_plot.png'):
    # 根据 pseudo_labels 选择特征
    def select_expert_features(features, pseudo_labels):
        selected_features = []
        for i in range(len(pseudo_labels)):
            expert_index = pseudo_labels[i]
            selected_features.append(features[i, expert_index])
        return np.array(selected_features)
    
    # 选择训练集和验证集中的特征
    train_selected_features = select_expert_features(train_features, train_pseudo_labels)
    val_selected_features = select_expert_features(val_features, val_pseudo_labels)
    
    # 将选中的特征合并
    all_features = np.concatenate([train_selected_features, val_selected_features])
    
    # 检查特征是否为空
    if all_features.shape[1] == 0:
        raise ValueError("The input features for PCA are empty.")
    
    # 进行PCA降维
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_features)
    
    # 分割PCA结果为训练集和验证集
    train_pca_results = pca_results[:len(train_selected_features)]
    val_pca_results = pca_results[len(train_selected_features):]
    
    # 定义颜色
    colors = ['#FF69B4', '#1E90FF']  # 粉色和蓝色
    
    # 绘制PCA图
    plt.figure(figsize=(10, 10))
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        # 绘制训练集数据点
        plt.scatter(train_pca_results[train_indices, 0], train_pca_results[train_indices, 1],
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx % len(colors)])
        
        # 绘制验证集数据点
        plt.scatter(val_pca_results[val_indices, 0], val_pca_results[val_indices, 1],
                    label=f'Val Class {class_idx}', alpha=0.8, edgecolor='k', color=colors[class_idx % len(colors)])
    
    # 添加图例、标题和标签
    plt.legend()
    plt.title("PCA of Selected Expert Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    
    # 保存并显示图像
    plt.savefig(filename)
    plt.show()

def plot_all_experts_select_feat_tsne(train_features, train_labels, val_features, val_labels, train_pseudo_labels, val_pseudo_labels, num_classes, filename='tsne_plot.png'):
    # 定义颜色
    colors = ["#5285c6", "#3fa0c0", "#4c6c43", "#d6e0c8", "#b55489", "#f1a19a"]
    # pseudo_labels_tensor = torch.from_numpy(train_pseudo_labels)

    # 获取类别数量
    # num_classes = pseudo_labels_tensor.max().item() + 1

    # # 确保索引张量是长整型
    # pseudo_labels_tensor = pseudo_labels_tensor.long()

    # # 进行 one-hot 编码
    # pseudo_labels_one_hot = F.one_hot(pseudo_labels_tensor, num_classes=num_classes).float()

    # # 如果需要转换回 NumPy 数组
    # pseudo_labels_one_hot = pseudo_labels_one_hot.numpy()
    # 初始化 t-SNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000)
    
    # 根据伪标签选择相应专家的特征
    # 将 NumPy 数组转换为 PyTorch 张量
    train_features_tensor = torch.from_numpy(train_features)
    val_features_tensor = torch.from_numpy(val_features)

    # 选择训练集和验证集的特征
    selected_train_features = torch.stack([train_features_tensor[i, train_pseudo_labels[i], :] for i in range(train_features_tensor.shape[0])], dim=0)
    selected_val_features = torch.stack([val_features_tensor[i, val_pseudo_labels[i], :] for i in range(val_features_tensor.shape[0])], dim=0)

    # 将训练集和验证集特征合并
    all_features = torch.cat([selected_train_features, selected_val_features], dim=0)

    # 进行 t-SNE 降维
    tsne_result = tsne.fit_transform(all_features.cpu().numpy())

    
    # 分割 t-SNE 结果为训练集和验证集
    train_tsne_results = tsne_result[:len(selected_train_features)]
    val_tsne_results = tsne_result[len(selected_train_features):]
    
    plt.figure(figsize=(10, 10))
    
    # 绘制 t-SNE 图，训练集无边框，验证集有边框
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        # 训练集
        plt.scatter(train_tsne_results[train_indices, 0], train_tsne_results[train_indices, 1],
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx], edgecolor='none')
        
        # 验证集
        plt.scatter(val_tsne_results[val_indices, 0], val_tsne_results[val_indices, 1],
                    label=f'Val Class {class_idx}', alpha=0.8, color=colors[class_idx], edgecolor='k')
    
    plt.legend()
    plt.title("t-SNE of Features Selected by Pseudo Labels")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
def plot_all_experts_select_feat_pca(train_features, train_labels, val_features, val_labels, train_pseudo_labels, val_pseudo_labels,  num_classes, filename='pca_plot.png'):
    # 定义颜色
    colors = [
          "#8ecfc9","#ffbe7a","#fa7f6f","#82b0d2","#beb8dc","#e7dad2"
        ]
    # 初始化 PCA
    pca = PCA(n_components=2)
    # 将 NumPy 数组转换为 PyTorch 张量
    train_features_tensor = torch.from_numpy(train_features)
    val_features_tensor = torch.from_numpy(val_features)

    # 选择训练集和验证集的特征
    selected_train_features = torch.stack([train_features_tensor[i, train_pseudo_labels[i], :] for i in range(train_features_tensor.shape[0])], dim=0)
    selected_val_features = torch.stack([val_features_tensor[i, val_pseudo_labels[i], :] for i in range(val_features_tensor.shape[0])], dim=0)
    
    # 将训练集和验证集特征合并
    all_features = torch.cat([selected_train_features, selected_val_features], dim=0)
    
    # 进行 PCA 降维
    pca_result = pca.fit_transform(all_features.cpu().numpy())
    
    # 分割 PCA 结果为训练集和验证集
    train_pca_results = pca_result[:len(selected_train_features)]
    val_pca_results = pca_result[len(selected_train_features):]
    
    plt.figure(figsize=(10, 10))
    
    # 绘制 PCA 图，训练集无边框，验证集有边框
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        # 训练集
        plt.scatter(train_pca_results[train_indices, 0], train_pca_results[train_indices, 1],
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx], edgecolor='none')
        
        # 验证集
        plt.scatter(val_pca_results[val_indices, 0], val_pca_results[val_indices, 1],
                    label=f'Val Class {class_idx}', alpha=0.8, color=colors[class_idx], edgecolor='k')
    
    plt.legend()
    plt.title("PCA of Features Selected by Pseudo Labels")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_all_experts_tsne(train_features, train_labels, val_features, val_labels, num_classes, experts_range, filename='tsne_plot.png'):
    # Generate a colormap for different experts
    num_experts = len(experts_range)
    cmap = plt.get_cmap('tab10')  # Use a colormap that supports many colors
    # colors = [cmap(i) for i in np.linspace(0, 1, num_experts)]  # Generate colors for each expert
    colors = [
          "#5285c6","#3fa0c0","#4c6c43","#d6e0c8","#b55489","#f1a19a"
        ]
    # Combine training and validation features
    all_features = [np.concatenate([train_features[:, expert_idx, :], val_features[:, expert_idx, :]]) for expert_idx in experts_range]
    
    # Initialize TSNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000)
    
    # Transform all features using TSNE
    tsne_results = [tsne.fit_transform(features) for features in all_features]
    
    plt.figure(figsize=(10, 10))
    
    for expert_idx, tsne_result in enumerate(tsne_results):
        train_tsne_results = tsne_result[:len(train_features)]
        val_tsne_results = tsne_result[len(train_features):]
        
        for class_idx in range(num_classes):
            train_indices = train_labels == class_idx
            val_indices = val_labels == class_idx
            
            plt.scatter(train_tsne_results[train_indices, 0], train_tsne_results[train_indices, 1],
                        label=f'Train Class {class_idx} (Expert {expert_idx})', alpha=0.5, color=colors[expert_idx*num_classes+class_idx])
            
            plt.scatter(val_tsne_results[val_indices, 0], val_tsne_results[val_indices, 1],
                        label=f'Val Class {class_idx} (Expert {expert_idx})', alpha=0.8, edgecolor='k', color=colors[expert_idx*num_classes+class_idx])
    
    plt.legend()
    plt.title("t-SNE of Features from All Experts")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_all_experts_pca(train_features, train_labels, val_features, val_labels, num_classes, experts_range, filename='pca_plot.png'):
    # Generate a colormap for different experts
    num_experts = len(experts_range)
    cmap = plt.get_cmap('tab10')  # Use a colormap that supports many colors
    # colors = [cmap(i) for i in np.linspace(0, 1, num_experts)]  # Generate colors for each expert
    colors = [
          "#5285c6","#3fa0c0","#4c6c43","#d6e0c8","#b55489","#f1a19a"
        ]
    
    # Combine training and validation features for each expert
    all_features = [np.concatenate([train_features[:, expert_idx, :], val_features[:, expert_idx, :]]) for expert_idx in experts_range]
    
    # Initialize PCA
    pca = PCA(n_components=2)
    
    # Transform all features using PCA
    pca_results = [pca.fit_transform(features) for features in all_features]
    
    plt.figure(figsize=(10, 10))
    
    for expert_idx, pca_result in enumerate(pca_results):
        train_pca_results = pca_result[:len(train_features)]
        val_pca_results = pca_result[len(train_features):]
        
        for class_idx in range(num_classes):
            train_indices = train_labels == class_idx
            val_indices = val_labels == class_idx
            
            plt.scatter(train_pca_results[train_indices, 0], train_pca_results[train_indices, 1],
                        label=f'Train Class {class_idx} (Expert {expert_idx})', alpha=0.5, color=colors[expert_idx*num_classes+class_idx])
            
            plt.scatter(val_pca_results[val_indices, 0], val_pca_results[val_indices, 1],
                        label=f'Val Class {class_idx} (Expert {expert_idx})', alpha=0.8, edgecolor='k', color=colors[expert_idx*num_classes+class_idx])
    
    plt.legend()
    plt.title("PCA of Features from All Experts")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
def plot_tsne(train_features, train_labels, val_features, val_labels, num_classes, filename='tsne_plot.png'):
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000)
    all_features = np.concatenate([train_features, val_features])
    
    # Check if all_features is valid
    if all_features.shape[1] == 0:
        raise ValueError("The input features for TSNE are empty.")
    
    tsne_results = tsne.fit_transform(all_features)
    
    train_tsne_results = tsne_results[:len(train_features)]
    val_tsne_results = tsne_results[len(train_features):]
    
    # Define the colors: pink and blue
    colors = ['#FF69B4', '#1E90FF']  # Pink and blue

    plt.figure(figsize=(10, 10))
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        plt.scatter(train_tsne_results[train_indices, 0], train_tsne_results[train_indices, 1], 
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx % len(colors)])
        
        plt.scatter(val_tsne_results[val_indices, 0], val_tsne_results[val_indices, 1], 
                    label=f'Val Class {class_idx}', alpha=0.8, edgecolor='k', color=colors[class_idx % len(colors)])
    
    plt.legend()
    plt.title("t-SNE of Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def extract_features(loader, model, args):
    all_features = None  # 初始化为 None
    all_labels = []
    all_predictions = []
    all_pseudo_labels = None  # 初始化为 None
    
    with torch.no_grad():
        for i, (images, clinic_feature, manual_features, target, idx) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            
            with autocast(enabled=args.amp):
                if args.model_type in ["ClinicalImageBranchClusterModel", "ClinicalImageBranchClusterModel_Lv2", "ClinicalImageBranchClusterModel_Lv3", "ClinicalImageBranchClusterGatingModel"]:
                    outputs = model(images, clinic_feature, epoch=args.epochs-1)
                elif args.input_radiomic:
                    outputs = model(images, clinic_feature,manual_features)
                elif args.model_type in ["GCALBEF"]:
                    outputs = model(images, clinic_feature,mode = 'val')
                else:
                    outputs = model(images, clinic_feature)
                
                # 检查 'features' 是否存在
                if 'features' in outputs:
                    features = outputs['features'].detach().cpu().numpy()
                    if all_features is None:
                        all_features = []  # 如果是第一次遇到特征，初始化列表
                    all_features.append(features)
                
                # 获取 predictions 并转换为 NumPy 数组
                predictions = F.softmax(outputs['output'].detach().cpu(), dim=1).numpy()
                all_predictions.append(predictions)
                
                # 检查 'pseudo_labels' 是否存在
                if 'pseudo_labels' in outputs:
                    pseudo_labels = outputs['pseudo_labels'].cpu().numpy()
                    if all_pseudo_labels is None:
                        all_pseudo_labels = []  # 如果是第一次遇到伪标签，初始化列表
                    all_pseudo_labels.append(pseudo_labels)
                
                # 添加 labels
                all_labels.append(target.detach().cpu().numpy())
    
    # 将列表转换为 NumPy 数组或保持为 None
    if all_features is not None:
        all_features = np.vstack(all_features)
    if all_pseudo_labels is not None:
        all_pseudo_labels = np.concatenate(all_pseudo_labels)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    return all_features, all_labels, all_predictions, all_pseudo_labels
def js_divergence(p, q):
    # 确保概率分布的总和为1
    p = torch.clamp(p, 1e-15, 1 - 1e-15)
    q = torch.clamp(q, 1e-15, 1 - 1e-15)
    m = 0.5 * (p + q)
    js = 0.5 * (torch.sum(p * torch.log(p / m), dim=1) + torch.sum(q * torch.log(q / m), dim=1))
    return js
def plot_losses(train_losses, val_losses1, val_losses2, output_path):
    """
    绘制训练集和两个测试集（验证集和EMA验证集）的损失曲线，并保存为图片。

    Args:
    - train_losses (list): 训练集损失列表
    - val_losses1 (list): 验证集1损失列表
    - val_losses2 (list): EMA验证集损失列表
    - output_path (str): 图片保存路径
    """

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)

    # 绘制训练集损失图
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'train_loss_curve.png'))
    plt.show()

    # 绘制验证集1损失图
    plt.figure()
    plt.plot(range(1, len(val_losses1) + 1), val_losses1, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'val_loss_curve1.png'))
    plt.show()

    # 绘制EMA验证集损失图
    plt.figure()
    plt.plot(range(1, len(val_losses2) + 1), val_losses2, label='EMA Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('EMA Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'val_loss_curve2.png'))
    plt.show()
def generate_pseudo_labels(predictions, true_labels, num_experts):
    batch_size, num_experts, num_classes = predictions.shape
    pseudo_labels = torch.zeros(batch_size, num_experts, device=predictions.device)
    
    # 预测值.detach()确保在计算距离时不会产生梯度
    detached_predictions = predictions.detach()
    
    # 计算每个专家的预测值与真实标签的距离
    distances = torch.zeros(batch_size, num_experts)
    for i in range(num_experts):
        pred = detached_predictions[:, i, :]  # 单个专家的预测值
        # 使用交叉熵损失作为距离度量
        distances[:, i] = F.cross_entropy(pred, true_labels, reduction='none')
    
    # 选择与真实标签距离最小的两个专家
    _, top2_indices = torch.topk(-distances, 2, dim=1)
    top2_indices = top2_indices.to(pseudo_labels.device)
    # 生成伪标签
    pseudo_labels.scatter_(1, top2_indices, 1)
    
    return pseudo_labels
def average_js_divergence_among_experts(model_output):
    """
    计算模型输出中各个专家之间的Jensen-Shannon散度的平均值。
    
    参数:
    model_output -- 模型的输出张量，形状为 [batch_size, num_experts, num_classes]
    
    返回:
    average_js -- 各个专家输出之间的JS散度的平均值
    """
    batch_size, num_experts, num_classes = model_output.shape
    # 沿着dim=1拆分张量
    expert_outputs = model_output.view(batch_size * num_experts, num_classes)
    
    # 归一化每个专家的输出
    expert_outputs /= torch.sum(expert_outputs, dim=1, keepdim=True)
    
    # 计算所有专家输出对之间的JS散度
    js_values = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            idx_i = i * batch_size
            idx_j = j * batch_size
            js_values.append(js_divergence(
                expert_outputs[idx_i:idx_i+batch_size],
                expert_outputs[idx_j:idx_j+batch_size]
            ))
    
    # 计算平均JS散度
    average_js = torch.mean(torch.stack(js_values))
    return average_js

def plot_multiclass_roc_curve(targets, all_outputs, method, fold, output):
    plt.figure(figsize=(12, 8))
    n_experts = all_outputs.shape[1]
    n_classes = all_outputs.shape[2]
    
    for expert_idx in range(n_experts):
        for class_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(targets, all_outputs[:, expert_idx, class_idx], pos_label=class_idx)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}) for Expert {expert_idx + 1}, Class {class_idx}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method} - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output, f'ROC_{method}_fold_{fold}.png'))
    plt.show()

def plot_multiclass_pr_curve(targets, all_outputs, method, fold, output):
    plt.figure(figsize=(12, 8))
    n_experts = all_outputs.shape[1]
    n_classes = all_outputs.shape[2]
    
    for expert_idx in range(n_experts):
        for class_idx in range(n_classes):
            precision, recall, _ = precision_recall_curve(targets, all_outputs[:, expert_idx, class_idx], pos_label=class_idx)
            plt.plot(recall, precision, label=f'PR Curve for Expert {expert_idx + 1}, Class {class_idx}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {method} - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output, f'PR_{method}_fold_{fold}.png'))
    plt.show()
def plot_dca_curves(targets, dl_outputs, filename):
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit = np.zeros(len(thresholds))
    
    for j, threshold in enumerate(thresholds):
        y_pred = (dl_outputs[:, 1] >= threshold).astype(int)  # 假设二分类模型
        tn, fp, fn, tp = confusion_matrix(targets, y_pred).ravel()
        
        n = len(targets)
        net_benefit[j] = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, net_benefit, label='Net Benefit')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_roc_curve(y_true, y_scores, num_classes, filename):
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
# PCA降维
def pca_base_plot(train_features, train_labels, train_pseudo_labels, val_features, val_labels, val_pseudo_labels,filename):
    # 合并训练集和验证集特征
    all_features = np.vstack((train_features, val_features))
    
    # 使用PCA降维至2D
    pca = PCA(n_components=2)
    all_features_2d = pca.fit_transform(all_features)
    
    # 分开训练集和验证集的降维结果
    train_features_2d = all_features_2d[:len(train_features)]
    val_features_2d = all_features_2d[len(train_features):]
    
    # 定义颜色映射，伪标签决定颜色，不同label决定深浅
    colors = ['red', 'green', 'blue']  # 3个伪标签类别分别对应不同的颜色
    color_map = {0: 0.4, 1: 0.8}  # label为0时颜色浅，label为1时颜色深

    plt.figure(figsize=(10, 7))
    
    # 绘制训练集点（无黑边）
    for pseudo_label in range(3):  # 遍历伪标签
        idx = train_pseudo_labels == pseudo_label
        for label in [0, 1]:  # 遍历label
            label_idx = train_labels == label
            final_idx = idx & label_idx
            plt.scatter(
                train_features_2d[final_idx, 0], 
                train_features_2d[final_idx, 1], 
                color=colors[pseudo_label],
                alpha=color_map[label],  # 透明度决定颜色深浅
                edgecolors='none',  # 无边框
                label=f'Train: pseudo_label {pseudo_label}, label {label}' if label == 0 else ""  # 防止重复label
            )
    
    # 绘制验证集点（有黑边）
    for pseudo_label in range(3):  # 遍历伪标签
        idx = val_pseudo_labels == pseudo_label
        for label in [0, 1]:  # 遍历label
            label_idx = val_labels == label
            final_idx = idx & label_idx
            plt.scatter(
                val_features_2d[final_idx, 0], 
                val_features_2d[final_idx, 1], 
                color=colors[pseudo_label],
                alpha=color_map[label],  # 透明度决定颜色深浅
                edgecolors='black',  # 黑色边框
                linewidth=1,
                label=f'Val: pseudo_label {pseudo_label}, label {label}' if label == 0 else ""
            )

    # 设置图例和标题
    plt.title("PCA Visualization of Train and Val Features")
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.show()
# T-SNE降维
def tsne_base_plot(train_features, train_labels, train_pseudo_labels, val_features, val_labels, val_pseudo_labels,filename):
    # 合并训练集和验证集特征
    all_features = np.vstack((train_features, val_features))
    
    # 使用T-SNE降维至2D
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)
    
    # 分开训练集和验证集的降维结果
    train_features_2d = all_features_2d[:len(train_features)]
    val_features_2d = all_features_2d[len(train_features):]
    
    # 定义颜色映射，伪标签决定颜色，不同label决定深浅
    colors = ['red', 'green', 'blue']
    color_map = {0: 0.4, 1: 0.8}  # label为0时颜色浅，label为1时颜色深

    plt.figure(figsize=(10, 7))
    
    # 绘制训练集点（无黑边）
    for pseudo_label in range(3):
        idx = train_pseudo_labels == pseudo_label
        for label in [0, 1]:
            label_idx = train_labels == label
            final_idx = idx & label_idx
            plt.scatter(
                train_features_2d[final_idx, 0], 
                train_features_2d[final_idx, 1], 
                color=colors[pseudo_label],
                alpha=color_map[label], 
                edgecolors='none', 
                label=f'Train: pseudo_label {pseudo_label}, label {label}' if label == 0 else "" #防止重复label
            )
    
    # 绘制验证集点（有黑边）
    for pseudo_label in range(3):
        idx = val_pseudo_labels == pseudo_label
        for label in [0, 1]:
            label_idx = val_labels == label
            final_idx = idx & label_idx
            plt.scatter(
                val_features_2d[final_idx, 0], 
                val_features_2d[final_idx, 1], 
                color=colors[pseudo_label],
                alpha=color_map[label], 
                edgecolors='black', 
                linewidth=1,
                label=f'Val: pseudo_label {pseudo_label}, label {label}' if label == 0 else ""
            )

    # 设置图例和标题
    plt.title("T-SNE Visualization of Train and Val Features")
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.show()
def plot_pr_curve(y_true, y_scores, num_classes, filename):
    plt.figure()
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {average_precision:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.show()
def one_hot(labels, num_classes=None):
    """
    将标签转换为独热编码
    :param labels: 标签，可以是list、tuple、ndarray等
    :param num_classes: 标签总数，如果不指定则根据labels中的值自动确定
    :return: 独热编码矩阵
    """
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]
def calculate_binary_metrics(targets, dl_outputs, threshold=0.5):
    logits = dl_outputs
    # Convert logits to probabilities using sigmoid for binary classification
    probabilities = F.softmax(torch.tensor(logits).float(), dim=1).detach().cpu().numpy()
    acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(targets,probabilities)
    # Convert logits to probabilities using sigmoid for binary classification
    metrics = {
        'AUC': auc,
        'Accuracy': acc,
        'CI': ci,
        'TPR': tpr,
        'TNR': tnr,
        'PPV': ppv,
        'NPV': npv,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Threshold': thres,
    }
    # predictions = (probabilities[:, 1] >= threshold).astype(int)
    
    # acc_score = accuracy_score(targets, predictions)
    # pre_score, rec_score, f1_score, _ = precision_recall_fscore_support(targets, predictions, average='binary')
    # conf_matrix = confusion_matrix(targets, predictions)
    
    # # Specificity calculation for binary classification
    # TN = conf_matrix[0, 0]
    # FP = conf_matrix[0, 1]
    # specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # # Calculate AUC for binary classification
    # fpr, tpr, _ = roc_curve(targets, probabilities[:, 1], pos_label=1)
    # auc_value = auc(fpr, tpr)
    
    # metrics = {
    #     'Accuracy': acc_score,
    #     'Precision': pre_score,
    #     'Recall': rec_score,
    #     'F1-Score': f1_score,
    #     'Specificity': specificity,
    #     'AUC': auc_value
    # }

    return metrics

def log_and_save_binary_metrics(args, metrics, mode, epoch, output_class,loss=None, logger=None, threshold=0.5):
    # eval_metrics = []
    
    if loss is None:
        loss = 0.0
        
    # Prepare the average metrics entry
    metrics = {
        'Epoch': epoch,
        'Fold': args.fold,
        'Method': output_class,
        'Mode': mode,
        'AUC': metrics['AUC'],
        'Accuracy': metrics['Accuracy'],
        'CI': metrics['CI'],
        'TPR': metrics['TPR'],
        'TNR': metrics['TNR'],
        'PPV': metrics['PPV'],
        'NPV': metrics['NPV'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score'],  # Add Specificity to the metrics
        'Threshold': metrics['Threshold'],
        'Loss': loss
        # 'Confusion Matrix': metric['Confusion Matrix']
    }

    # Convert the metrics to a DataFrame
    df_metrics = pd.DataFrame([metrics])
    logger.info(f"Epoch {epoch} Metrics: Accuracy: {metrics['Accuracy']}, PPV: {metrics['PPV']}, NPV: {metrics['NPV']}, CI: {metrics['CI']}, TPR: {metrics['TPR']}, TNR: {metrics['TNR']}, AUC: {metrics['AUC']}, threshold: {metrics['Threshold']}")

    # Define the path to the CSV file
    csv_file = os.path.join(args.output, args.excel_file_name)

    # Save the DataFrame to CSV (append mode if file exists)
    if os.path.exists(csv_file):
        df_metrics.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(csv_file, index=False)
def calculate_multiclass_metrics(targets, dl_outputs, final_output=None):
    num_experts = dl_outputs.shape[1]  # Number of experts

    metrics = []

    # Compute metrics for each head
    for i in range(num_experts):
        head_outputs = dl_outputs[:, i, 1]
        # auc_scores = roc_auc_score(targets, head_outputs)
        # acc_scores = accuracy_score(targets, np.where(head_outputs > 0.5, 1, 0))
        # pre_scores = precision_score(targets, np.where(head_outputs > 0.5, 1, 0))
        # rec_scores = recall_score(targets, np.where(head_outputs > 0.5, 1, 0))
        # f1_scores = f1_score(targets, np.where(head_outputs > 0.5, 1, 0))
        # conf_matrix = confusion_matrix(targets, np.where(head_outputs > 0.5, 1, 0))
        # specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        # ap_score = average_precision_score(targets, head_outputs)
        acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(targets,head_outputs)
        metrics.append({
            'AUC': auc,
            'Accuracy': acc,
            'CI': ci,
            'TPR': tpr,
            'TNR': tnr,
            'PPV': ppv,
            'NPV': npv,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Threshold': thres,
        })

    # Compute average outputs
    if final_output is not None:
        avg_outputs = final_output[:,1]
        output_avg_outputs = final_output
    else:
        avg_outputs = np.mean(dl_outputs[:,:, 1], axis=1)
        output_avg_outputs = np.mean(dl_outputs, axis=1)
     
    # Compute metrics for the average outputs
    # avg_auc_scores = roc_auc_score(targets, avg_outputs)
    # avg_acc_scores = accuracy_score(targets, np.where(avg_outputs > 0.5, 1, 0))
    # avg_pre_scores = precision_score(targets, np.where(avg_outputs > 0.5, 1, 0))
    # avg_rec_scores = recall_score(targets, np.where(avg_outputs > 0.5, 1, 0))
    # avg_f1_scores = f1_score(targets, np.where(avg_outputs > 0.5, 1, 0))
    # avg_conf_matrix = confusion_matrix(targets, np.where(avg_outputs > 0.5, 1, 0))
    # avg_specificity = avg_conf_matrix[0, 0] / (avg_conf_matrix[0, 0] + avg_conf_matrix[0, 1])
    # avg_ap_score = average_precision_score(targets, avg_outputs)
    acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(targets,avg_outputs)

    avg_metrics = {
            'AUC': auc,
            'Accuracy': acc,
            'CI': ci,
            'TPR': tpr,
            'TNR': tnr,
            'PPV': ppv,
            'NPV': npv,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Threshold': thres,
    }

    return metrics, avg_metrics, output_avg_outputs
# import os
# import pandas as pd

def log_and_save_metrics(args, metrics, avg_metrics, logger, mode, epoch,loss=None, threshold=0.5):
    # Initialize list to store evaluation metrics
    eval_metrics = []
    # Save metrics for each expert
    if loss == None:
        loss = 0.0
    if metrics:
        for i, metric in enumerate(metrics):
            expert_metrics = {
                'Epoch': epoch,
                'Expert': f"Expert {i + 1}",
                'Fold': args.fold + 1,
                'Method': args.logit_method,
                'Mode': mode,
                'AUC': metric['AUC'],
                'Accuracy': metric['Accuracy'],
                'CI': metric['CI'],
                'TPR': metric['TPR'],
                'TNR': metric['TNR'],
                'PPV': metric['PPV'],
                'NPV': metric['NPV'],
                'Precision': metric['Precision'],
                'Recall': metric['Recall'],
                'F1-Score': metric['F1-Score'],  # Add Specificity to the metrics
                'Threshold': metric['Threshold'],
                'Loss': loss
                # 'Confusion Matrix': metric['Confusion Matrix']
            }
            eval_metrics.append(expert_metrics)

            # Log expert metrics
            logger.info(f"Epoch {epoch} - Expert {i + 1} Metrics: Accuracy: {metric['Accuracy']}, PPV: {metric['PPV']}, NPV: {metric['NPV']}, CI: {metric['CI']}, TPR: {metric['TPR']}, TNR: {metric['TNR']}, AUC: {metric['AUC']}, threshold: {metric['Threshold']}")

    # Save average metrics
    avg_metrics_entry = {
        'Epoch': epoch,
        'Expert': 'Average',
        'Fold': args.fold + 1,
        'Method': args.logit_method,
        'Mode': mode,
        'AUC': avg_metrics['AUC'],
        'Accuracy': avg_metrics['Accuracy'],
        'CI': avg_metrics['CI'],
        'TPR': avg_metrics['TPR'],
        'TNR': avg_metrics['TNR'],
        'PPV': avg_metrics['PPV'],
        'NPV': avg_metrics['NPV'],
        'Precision': avg_metrics['Precision'],
        'Recall': avg_metrics['Recall'],
        'F1-Score': avg_metrics['F1-Score'],  # Add Specificity to the metrics
        'Threshold': avg_metrics['Threshold'],
        'Loss': loss,
        'threshold': threshold
        # 'Confusion Matrix': avg_metrics['Confusion Matrix']
    }
    eval_metrics.append(avg_metrics_entry)

    # Log average metrics
    logger.info(f"Epoch {epoch} - Average Metrics: Accuracy: {avg_metrics['Accuracy']}, PPV: {avg_metrics['PPV']}, NPV: {avg_metrics['NPV']}, CI: {avg_metrics['CI']}, TPR: {avg_metrics['TPR']}, TNR: {avg_metrics['TNR']}, AUC: {avg_metrics['AUC']}, threshold: {avg_metrics['Threshold']}")

    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(eval_metrics)

    # Define CSV file path
    csv_file = os.path.join(args.output, args.excel_file_name)

    # Save to CSV
    if os.path.exists(csv_file):
        df_metrics.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(csv_file, index=False)

def calculate_net_benefit(targets, predictions, threshold):
    # Convert predictions to binary decisions based on threshold
    decisions = np.where(predictions >= threshold, 1, 0)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((decisions == 1) & (targets == 1))
    fp = np.sum((decisions == 1) & (targets == 0))
    tn = np.sum((decisions == 0) & (targets == 0))
    fn = np.sum((decisions == 0) & (targets == 1))
    
    # Calculate net benefit
    net_benefit = (tp - fp) / len(targets)
    
    return net_benefit

##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')

def sigmoid_increase(current_step, total_steps, final_value=0.2, k=5):
    if current_step < 0 or current_step >= total_steps:
        raise ValueError("current_step 应该在 [0, total_steps) 范围内")

    x = (2 * current_step / total_steps) - 1
    return final_value / (1 + np.exp(-k * x))
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist
import numpy as np

def multi_class_mAP(imagessetfilelist, num_classes, return_each=False):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    
    aps = np.zeros(num_classes)
    mAPs = []

    for class_id in range(num_classes):
        lines = []
        for imagessetfile in imagessetfilelist:
            with open(imagessetfile, 'r') as f:
                lines.extend(f.readlines())

        seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
        gt_label = seg[:, class_id].astype(np.int32)  # Extract ground truth labels for the current class
        num_target = np.sum(gt_label)

        sample_num = len(gt_label)
        tp = np.zeros(sample_num)
        fp = np.zeros(sample_num)

        confidence = seg[:, class_id]  # Extract confidence scores for the current class
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(num_target)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        
        # Calculate Average Precision (AP) for the current class
        ap = voc_ap(rec, prec, num_target)  # You need to define or implement voc_ap function
        aps[class_id] = ap * 100  # Store AP in percentage
        
        if return_each:
            mAPs.append(aps[class_id])

    mAP = np.mean(aps)
    
    if return_each:
        return mAP, mAPs
    
    return mAP

def voc_ap(rec, prec, num_target):
    """
    Calculate Average Precision (AP) from precision-recall curve.
    This function computes the VOC 2007 challenge AP given precision and recall.
    Args:
        rec: Array of recall values.
        prec: Array of precision values.
        num_target: Number of positive samples (targets).
    Returns:
        Average Precision (AP) for the given precision-recall curve.
    """
    rec = np.concatenate(([0.], rec, [1.]))
    prec = np.concatenate(([0.], prec, [0.]))

    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    inds = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[inds + 1] - rec[inds]) * prec[inds + 1])  # VOC 2007 AP

    return ap

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict



if __name__ == '__main__':
    main()
    