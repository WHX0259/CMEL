import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
import os
from sklearn.preprocessing import label_binarize
def dice_coefficient(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() + smooth
    dice = (2. * intersection + smooth) / union
    return dice

def intersection_over_union(pred, target):
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = intersection / union
    return iou

# 计算评估指标的函数
def compute_metrics(targets, outputs):
    # 将输出的概率转换为预测标签
    # predictions = np.argmax(outputs, axis=1)
    predictions = np.where(outputs > 0.5, 1, 0)
    
    # 计算准确率
    accuracy = accuracy_score(targets, predictions)
    
    # 计算精确率、召回率和 F1 分数
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    # 计算 AUROC
    auroc = roc_auc_score(targets, outputs)  # 使用类别 1 的概率进行 AUROC 计算(352,)(352,2)
    
    # 计算特异度
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    specificity = tn / (tn + fp)
    
    return accuracy, precision, recall, f1, specificity, auroc


def compute_metrics_multi(targets, outputs):
    # 将输出的概率转换为预测标签
    predictions = np.argmax(outputs, axis=1)
    
    # 计算准确率
    accuracy = accuracy_score(targets, predictions)
    
    # 计算精确率、召回率和 F1 分数，使用 'macro' 平均
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')
    
    # 计算 AUROC，使用 'ovr'（一对多）模式
    auroc = roc_auc_score(targets, outputs, multi_class='ovr')
    
    # 计算特异度
    cm = confusion_matrix(targets, predictions)
    tn = np.diag(cm)
    fp = cm.sum(axis=0) - tn
    fn = cm.sum(axis=1) - tn
    tp = cm.sum() - (fp + fn + tn)
    specificity = np.mean(tn / (tn + fp))
    
    return accuracy, precision, recall, f1, specificity, auroc

def plot_roc_curve_multi(targets, outputs, epoch=0, auroc=None, output_dir=None, fold=0, mode='val'):
    num_classes = outputs.shape[1]

    # 将目标标签二值化
    targets_binarized = label_binarize(targets, classes=np.arange(num_classes))

    # 初始化字典存储 fpr, tpr 和 roc_auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 为每个类别计算 ROC 曲线和 AUC
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_binarized[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微观平均 ROC 曲线和 AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_binarized.ravel(), outputs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制所有类别的 ROC 曲线
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"], lw=2, linestyle=':', label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'roc_curve_epoch_{epoch}_fold_{fold}_{mode}.png'))
    plt.show()

# 绘制 ROC 曲线的函数
def plot_roc_curve(targets, outputs, epoch, auroc, output_dir,fold=0, mode = 'train'):
    fpr, tpr, _ = roc_curve(targets, outputs)  # 使用类别 1 的概率进行 ROC 曲线绘制
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (Area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend(loc="lower right")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'AUROC{}_epoch{}_{}.png'.format( auroc, epoch,mode)))

# Example usage:

# 示例用法：
# 假设 targets 和 outputs 是形状为 (num_samples,) 的 numpy 数组
# targets 应为 0 或 1，代表真实标签
# outputs 应为类别 1 的概率，形状为 (num_samples,)
# num_samples = 50
# targets = np.random.randint(0, 2, size=num_samples)  # 随机生成 0 或 1 的目标值
# outputs = np.random.rand(num_samples, 2)  # 随机生成形状为 (num_samples, 2) 的输出，表示类别 0 和 1 的概率

# # 调用评估指标函数和绘制 ROC 曲线函数
# accuracy, precision, recall, f1, specificity, auroc = compute_metrics(targets, outputs)
# plot_roc_curve(targets, outputs, epoch=1,auroc=auroc)
# # 打印评估指标
# print("准确率: {:.4f}".format(accuracy))
# print("精确率: {:.4f}".format(precision))
# print("召回率: {:.4f}".format(recall))
# print("F1 分数: {:.4f}".format(f1))
# print("特异度: {:.4f}".format(specificity))
# print("AUROC: {:.4f}".format(auroc))


# # Generate random mask and output
# output = torch.rand(10, 10)
# output_binary = (output > 0.5).float()  # Convert to binary
# target = torch.randint(0, 2, (10, 10)).float()

# # Compute Dice coefficient
# dice = dice_coefficient(output_binary, target)
# print("Dice Coefficient:", dice.item())

# # Compute Intersection over Union
# iou = intersection_over_union(output_binary, target)
# print("Intersection over Union:", iou.item())


# 测试代码
# if __name__ == "__main__":
#     # 创建模拟数据
#     num_samples = 100
#     num_classes = 10
#     np.random.seed(0)

#     # 随机生成真实标签和预测概率
#     targets = np.random.randint(0, num_classes, num_samples)#100,
#     outputs = np.random.rand(num_samples, num_classes)#100,10

#     # 使预测概率归一化
#     outputs = outputs / outputs.sum(axis=1, keepdims=True)

#     # 调用函数并绘制 ROC 曲线
#     plot_roc_curve_multi(targets, outputs, epoch=1, output_dir='.', fold=1, mode='val')
