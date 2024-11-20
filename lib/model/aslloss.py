"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()
import torch
import torch.nn.functional as F
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return torch.mean(entropy)
def diversity_loss(y_pred):
        # Calculate binary cross entropy loss
        y_pred_1 = y_pred
        y_pred_0 = 1 - y_pred
        entropy = Entropy(y_pred_1)+Entropy(y_pred_0)
        return entropy
class MixedLoss(torch.nn.Module):#混合损失
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, y_pred, y_true, weight=None):
        """
        Forward pass for mixed BCE Loss and Focal Loss
        :param y_pred: Predicted probabilities, shape [batch_size, num_classes]
        :param y_true: True labels, shape [batch_size]
        :return: Mixed Loss
        """
        # Calculate binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), weight=self.weight, reduction='none')
        
        # Calculate focal loss
        prob = torch.sigmoid(y_pred)
        focal_weights = self.alpha * (1 - prob)**self.gamma
        focal_loss = torch.mean(bce_loss * focal_weights)
        
        # Combine BCE loss and Focal loss
        mixed_loss = bce_loss + self.alpha * focal_loss
        
        if weight == None:
            loss = mixed_loss
            loss = loss.mean()
        else:
            loss = mixed_loss * weight
            loss = loss.sum() / torch.clip(weight.sum(), min=1e-8)
        return loss
class MixedLoss_ce(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(MixedLoss_ce, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, y_pred, y_true, weight=None):
        """
        Forward pass for mixed Cross Entropy Loss and Focal Loss
        :param y_pred: Predicted logits, shape [batch_size, num_classes]
        :param y_true: True labels, shape [batch_size] (integer class labels)
        :return: Mixed Loss
        """
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(y_pred, y_true, weight=self.weight, reduction='none')

        # Convert true labels to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        # Calculate the probability for the true class
        prob = F.softmax(y_pred, dim=1)
        prob_true_class = prob * y_true_one_hot
        prob_true_class = prob_true_class.sum(dim=1)

        # Calculate focal loss
        focal_weights = self.alpha * (1 - prob_true_class)**self.gamma
        focal_loss = torch.mean(ce_loss * focal_weights)

        # Combine CE loss and Focal loss
        mixed_loss = ce_loss + self.alpha * focal_loss

        if weight is None:
            loss = mixed_loss
            loss = loss.mean()
        else:
            loss = mixed_loss * weight
            loss = loss.sum() / torch.clip(weight.sum(), min=1e-8)
        return loss
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w

        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000
        return _loss
