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
from lib.utils.loss_71 import build_loss, Diversity_loss, MDCSLoss, RIDELoss
from lib.utils.metrics import analysis_pred_binary
from lib.model.ablationExperiment import ClinicalImageBaseClusterDistancePlusGatingModelC, ClinicalImageBaseClusterDistancePlusGatingModelB, ClinicalImageBaseClusterDistancePlusGatingModelA
from lib.model.ALBEF import GCALBEF
from lib.model.samms import CMEL_SammsNet
from lib.model.PathomicFusion import CMEL_PathomicNet
#改模型输出改损失

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    #下面是参数设置
    #主要调节下面这些变量
    parser.add_argument('--fold', type=int,default=3,
                        help="Name of the fold to use")
    parser.add_argument('--slice_path', type=str,default=r'/data/wuhuixuan/data/padding_crop',
                        help="Name of the fold to use")
    parser.add_argument('--fold_json', type=str,default=r'/data/huixuan/data/data_chi/TRG_patient_folds.json',
                        help="Name of the fold to use")
    parser.add_argument('--manual_csv_path', type=str,default=r'/data/wuhuixuan/code/Self_Distill_MoE/data/selected_features_22_with_id_label_fold_norm.csv',
                        help="Name of the fold to use")
    parser.add_argument('--sentence_json', type=str,default=r'/data/huixuan/code/Gastric_cancer_prediction/Gastric_cancer_predict/sentences.json',
                        help="Name of the fold to use")
    parser.add_argument('--csv_path', type=str,default=r'/data/huixuan/data/data_chi/label.csv',
                        help="Name of the fold to use")
    parser.add_argument('--criterion_type', type=str,default='mixed',
                        help="Name of the criterion to use")
    #下面这个是结果保存的位置，我保存为了csv文件extra_input
    parser.add_argument('--excel_file_name', help='note', default='/data/wuhuixuan/code/Self_Distill_MoE/result/MultiExperts_ASUS_Diversity_uskd.csv')
    parser.add_argument('--note', help='note', default='Causal experiment')
    parser.add_argument('--model_type',type=str, default = 'SE_ResNet50',help='note')
    parser.add_argument('--input_clinic',type=bool, default =False,help='note')
    #maybe you should change the output path
    parser.add_argument('--output', default='/data/wuhuixuan/code/Self_Distill_MoE/out')
    parser.add_argument('--num_class', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--logit_method', default='ResNeXt_Attention_Gating_Moe', type=str,
                        help="Number of query slots")
    parser.add_argument('--logit_alpha', default=0.5, type=float,
                        help="Number of query slots")
    parser.add_argument('--logit_beta', default=0.5, type=float,
                        help="Number of query slots")
    parser.add_argument('--logit_gamma', default=None, type=float,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',#模型预训练使用预训练的参数
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],#优化器
                        help='which optim to use')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    # loss
    parser.add_argument('--eps', default=1e-5, type=float,#防止为0，暂时没找到在哪用的
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--gamma', default=2.0, type=float,
                        metavar='gamma', help='gamma for focal loss')
    parser.add_argument('--alpha', default=0.25, type=float,
                        metavar='alpha', help='alpha for focal loss')
    parser.add_argument('--loss_dev', default=2, type=float,
                        help='scale factor for loss')#不知道，是把一些值变成负的
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',#epoch
                        help='number of total epochs to run')
    parser.add_argument('--focal_weight', type=float, default = 1.0, help='List of Loss weight')
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
    parser.add_argument('--clinic_dim', default=25, type=int,#extra_dim,临床信息的话维度是25，机器学习特征的话维度是22
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
def main():
    args = get_args()
    #调试添加参数
    args.excel_file_name = f'/data/wuhuixuan/code/Self_Distill_MoE/result/MultiExperts_{args.model_type}_{args.criterion_type}_Adjust_threshold.csv'
            
    if not os.path.exists(args.output):  # 如果路径不存在
        os.makedirs(args.output)

    args.output = os.path.join(args.output, args.model_type)
    if not os.path.exists(args.output):  # 加入模型类型
        os.makedirs(args.output)
    args.output = os.path.join(args.output, args.criterion_type)
    if not os.path.exists(args.output):  # 加入模型类型
        os.makedirs(args.output)
    args.output = os.path.join(args.output, str(args.fold + 1))
    if not os.path.exists(args.output):  # 加入文件夹的名称
        os.makedirs(args.output)
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
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

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

def calculate_similarity_scores(logits, targets):
    """
    计算并返回相似度分数。
    
    参数:
    logits -- 模型输出的 logits，形状为 [batch_size, num_experts, num_classes]
    targets -- 真实标签的索引形式，形状为 [batch_size]

    返回:
    similarity_scores -- 相似度分数，形状为 [batch_size, num_experts]
    """
    batch_size, num_experts, num_classes = logits.shape
    
    # 初始化相似度分数矩阵
    similarity_scores = torch.zeros(batch_size, num_experts, device=logits.device)
    
    # one-hot 编码目标标签
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
    
    # 计算每个专家的 softmax 概率分布和交叉熵损失
    for i in range(num_experts):
        # 对每个专家的 logits 应用 softmax
        softmax_probs = F.softmax(logits[:, i, :], dim=1)
        
        # 计算交叉熵损失，这里使用 reduction='none' 来获取每个样本的损失
        loss = F.cross_entropy(softmax_probs, one_hot_targets, reduction='none').view(batch_size)
        
        # 将损失转换为相似度分数，并累加到相似度分数矩阵中
        similarity_scores[:, i] = torch.exp(-loss)
    
    # 归一化相似度分数，使每个样本的专家相似度总和为 1
    similarity_scores = F.softmax(similarity_scores, dim=1)
    
    return similarity_scores
def extract_features(loader, model, args):
    all_features = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for i, (images, clinic, _, target, _) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with autocast(enabled=args.amp):
                if args.input_clinic:
                    outputs = model(images,clinic) 
                else:
                    outputs = model(images) 
                features = outputs['feature'].detach().cpu().numpy()
                predictions = outputs['output'].detach().cpu()
                predictions = F.softmax(predictions,dim=1)
                predictions = predictions.numpy()
            all_features.append(features)
            all_labels.append(target.detach().cpu().numpy())
            all_predictions.append(predictions)
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)#加权后的输出
    
    return all_features, all_labels, all_predictions
def main_worker(args, logger):
    global best_mAP
    global best_meanAUC
    global best_f1_score
    # args.resume=f"/data/wuhuixuan/code/Self_Distill_MoE/out/{args.model_type}/{args.criterion_type}/{args.fold+1}/train/{args.logit_method}/model_best.pth.tar"
    # Build model
    if args.model_type == 'ClinicalImageALBEFModel':
        model = ClinicalImageBaseClusterDistancePlusGatingModelA(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == 'ClinicalImageALBEFClusterModel':#ClinicalImageBranchClusterModel
        model = ClinicalImageBaseClusterDistancePlusGatingModelB(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == 'ClinicalImageALBEFClusterModel_Lv2':#ClinicalImageBranchClusterModel
        model = ClinicalImageBaseClusterDistancePlusGatingModelC(num_experts=args.num_experts, nlabels=args.num_class,num_iterations=5,cluster_init_type="kmeans++",k=8).cuda()
    elif args.model_type == 'GCALBEF':#ClinicalImageBranchClusterModel
        model = GCALBEF(clinic_length = args.clinic_dim, num_classes = args.num_class).cuda()

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    if args.criterion_type == "MDCSLoss":
        criterion = MDCSLoss(cls_num_list = args.cls_num_list)
    elif args.criterion_type == "RIDELoss":
        criterion = RIDELoss(cls_num_list = args.cls_num_list)
    else:
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
        metrics, net_benefit = test(train_loader, val_loader, model, args, logger, thresholds=0.5)
        logger.info(' Metrics: {}'.format(metrics))
        return
    # Criterion难道十折
    args.output = os.path.join(args.output, 'train')
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
import torch
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
    all_outputs = []  # Assuming three heads
    # extra_info = {}
    for i, (images, clinic, _, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
        # clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):

            if args.input_clinic:
                outputs = model(images,clinic) 
            else:
                outputs = model(images) 
            output = outputs["output"]
            feature = outputs["feature"]
            loss = 0.0
            loss = criterion(output, target)# output_logits, targets, extra_info=None, return_expert_losses=False# output_logits, targets, extra_info=None, return_expert_losses=False
            logger.info('loss: {:.3f}'.format(loss))

            if args.loss_dev > 0:
                loss *= args.loss_dev

        losses.update(loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        all_targets.append(target.detach().cpu().numpy())
        all_outputs.append(output.detach().cpu().numpy())
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

    metrics = calculate_binary_metrics(all_targets, all_outputs)
    log_and_save_metrics(args=args, metrics=metrics, mode='train', logger=logger,epoch=epoch, loss = losses.avg)
    return losses.avg, metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

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



def validate(val_loader, model, criterion, args, logger, epoch):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    model.eval()
    all_targets = []
    all_logits = []
    with torch.no_grad():
        end = time.time()
        for i, (images, clinic, _, target, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.input_clinic:
                    outputs = model(images,clinic) 
                else:
                    outputs = model(images) 
                output = outputs["output"]
                loss = criterion(output, target)# output_logits, targets, extra_info=None, return_expert_losses=False
            
                all_logits.append(output.detach().cpu().numpy())
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

        # Calculate metrics
        metrics = calculate_binary_metrics(all_targets, all_logits)
        ##(args, metrics, mode, epoch, loss=None, logger=None ,threshold=0.5)
        log_and_save_metrics(args=args, metrics=metrics, mode='val', logger=logger, epoch=epoch, loss=losses.avg)

        return losses.avg, metrics

        # return loss_avg, mAP, meanAUC, eval
import os
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger('MetricsLogger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

def log_and_save_metrics(args, metrics, mode, epoch, loss=None, logger=None, threshold=0.5):
    # eval_metrics = []
    
    if loss is None:
        loss = 0.0
        
    # Prepare the average metrics entry
    metrics = {
        'Epoch': epoch,
        'Fold': args.fold,
        'Method': args.logit_method,
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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, average_precision_score, auc
import torch.nn.functional as F
import os
import pickle
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
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

def test(train_loader, val_loader, model, args, logger, thresholds):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    best_threshold = 0.5
    best_thresholds = [0.5, 0.5]
    saved_data = []
    all_features = []
    all_targets = []
    all_outputs = []
    net_benefit = []
    
    # Load pre-trained models
    model_name_xgb = f'/data/wuhuixuan/code/Causal-main/save/XGBoost/xgb_model_fold_{args.fold + 1}.pkl'
    with open(model_name_xgb, 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    model_name_lgbm = f'/data/wuhuixuan/code/Causal-main/save/lgbm/lgbm_model_fold_{args.fold + 1}.pkl'
    with open(model_name_lgbm, 'rb') as model_file:
        lgbm_model = pickle.load(model_file)
    
    # Extract features and labels from train and validation sets
    train_features, train_labels, train_predictions = extract_features(train_loader, model, args)
    val_features, val_labels, _ = extract_features(val_loader, model, args)

    print("The best threshold is :" + str(best_threshold))

    # Load selected features for mapping
    selected_features_data = pd.read_csv('/data/wuhuixuan/code/Causal-main/data/selected_features_22_with_id_label_fold.csv')
    id_to_index = {str(row['ID']): idx for idx, row in selected_features_data.iterrows()}
    
    final_outputs = []
    with torch.no_grad():
        for i, (images, clinic, _, target, idx) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Forward pass
            with autocast(enabled=args.amp):
                if args.input_clinic:
                    outputs = model(images,clinic) 
                else:
                    outputs = model(images) 
            
            output = outputs["output"]  # Output from the model
            feature = outputs["feature"]  # Extracted features
            output_cpu = output.detach().cpu().numpy()

            all_outputs.append(output_cpu)
            all_targets.append(target.detach().cpu().numpy())

            batch_features = []
            mapped_labels = []
            
            # Mapping features based on index
            for idx_val in idx.numpy():
                feature_idx = id_to_index.get(str(idx_val) + '.nii.gz')
                if feature_idx is not None:
                    mapped_labels.append(selected_features_data.iloc[feature_idx]['label'])
                    batch_features.append(selected_features_data.drop(columns=['ID', 'label', 'fold']).iloc[feature_idx].values)
                else:
                    mapped_labels.append(None)
                    batch_features.append(np.zeros(selected_features_data.drop(columns=['ID', 'label', 'fold']).shape[1]))

            batch_features = np.array(batch_features)

            # Predict using XGBoost and LightGBM models
            xgb_predictions = xgb_model.predict_proba(batch_features)
            lgbm_predictions = lgbm_model.predict_proba(batch_features)

            # Combine outputs based on the selected method
            if args.logit_method == 'XGB':
                final_output = xgb_predictions
            elif args.logit_method == args.model_type:
                final_output = F.softmax(torch.tensor(output_cpu), dim=1).numpy()
            elif args.logit_method == 'LGBM':
                final_output = lgbm_predictions
            elif args.logit_method == 'XGB+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_beta * xgb_predictions) / (args.logit_alpha + args.logit_beta)
            elif args.logit_method == 'LGBM+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_gamma)
            elif args.logit_method == 'XGB+LGBM':
                final_output = (args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_beta + args.logit_gamma)
            elif args.logit_method == 'XGB+LGBM+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_beta + args.logit_gamma)

            final_outputs.append(final_output)
            final_output_class1 = final_output[:, 1]  # Probability of class 1

            for idx_val, final_prob, target_val in zip(idx.numpy(), final_output_class1, target.cpu().numpy()):
                saved_data.append([idx_val, target_val, final_prob])

    # Save the final output data to a CSV file
    saved_data = np.array(saved_data)
    columns = ['ID', 'Target', 'Avg_Output']
    df_saved_data = pd.DataFrame(saved_data, columns=columns)
    save_path = os.path.join(args.output, 'save_data.csv')
    df_saved_data.to_csv(save_path, index=False)

    # Concatenate outputs for evaluation
    all_final_outputs = np.concatenate(final_outputs, axis=0)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    # One-hot encoding for multi-class metrics
    all_target_onehot = one_hot(all_targets)

    # Calculate performance metrics
    metrics = calculate_binary_metrics(all_targets, all_final_outputs)

    # Plot ROC curve
    plot_roc_curve(all_target_onehot, all_final_outputs, num_classes=2, filename=os.path.join(args.output, 'roc_curve_avg_experts.png'))
    
    # Plot Precision-Recall curve
    plot_pr_curve(all_target_onehot, all_final_outputs, num_classes=2, filename=os.path.join(args.output, 'pr_curve_avg_experts.png'))
    
    # Plot Decision Curve Analysis (DCA)
    plot_dca_curves(all_targets, all_outputs, filename=os.path.join(args.output, f'dca_curves_expert_avg.png'))
    
    # Plot t-SNE
    plot_tsne(train_features, train_labels, val_features, val_labels, num_classes=2, filename=os.path.join(args.output, f'tsne_plot_expert.png'))

    # Log and save metrics
    log_and_save_metrics(args=args, metrics=metrics, mode='test', logger=logger, epoch=args.epochs, threshold=best_threshold)
    
    return metrics, net_benefit

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
            continue  # frozen weights
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


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict



if __name__ == '__main__':
    main()
    