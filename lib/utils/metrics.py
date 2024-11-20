# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/1/19
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
import sklearn.metrics as sm
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import column_or_1d, check_consistent_length, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
import scipy.stats


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma, logv=10):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
       logv:
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 10 ** (np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(logv)), z


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two, logv=10, with_z: bool = False):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
       logv:
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    pvalue, z = calc_pvalue(aucs, delongcov, logv=logv)
    if with_z:
        return pvalue, z
    else:
        return pvalue


def calc_95_CI(ground_truth, predictions, alpha=0.95, with_auc: bool = True):
    auc, auc_cov = delong_roc_variance(ground_truth, predictions)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    ci[ci < 0] = 0
    if with_auc:
        return auc, ci
    else:
        return ci

def calc_array_95ci(data, confidence=0.95):
    data = column_or_1d(np.array(data))
    std = stats.tstd(data)
    sem = stats.sem(data)
    return stats.t.interval(confidence, df=len(data) - 1, loc=np.mean(data), scale=sem)


def calc_value_95ci(a, b=None, sample_num=None) -> tuple:
    """
    实现： Wilson, E. B. "Probable Inference, the Law of Succession, and Statistical Inference,"
          Journal of the American Statistical Association, 22, 209-212 (1927).

    Args:
        a: 分子
        b: 分母
        sample_num: 样本数

    Returns: 95% CI [lower, upper]

    """
    if b is None:
        a = a * sample_num
        b = sample_num - a
    sum_value = a + b + 1e-6
    ratio = a / sum_value
    std = (ratio * (1 - ratio) / sum_value) ** 0.5
    return max(0, ratio - 1.96 * std), min(ratio + 1.96 * std, 1)


def map_ci(ci):
    ci_float = [float(f"{i_:.6f}") for i_ in ci]
    ci_float[0] = ci_float[0] if not np.isnan(ci_float[0]) else 1
    ci_float[1] = ci_float[1] if not np.isnan(ci_float[1]) else 1
    # print(ci_float)
    return ci_float


def check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    pos_label : int or str, default=None
        The label of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    tns = fps[-1] - fps
    fns = tps[-1] - tps
    return fps, tps, tns, fns, y_score[threshold_idxs]


def any_curve(y_true, y_score, *, pos_label=None, sample_weight=None):
    fps, tps, tns, fns, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    if tns[0] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "true negative value should be meaningless",
                      UndefinedMetricWarning)
        tnr = np.repeat(np.nan, tns.shape)
    else:
        tnr = tns / tns[0]

    if fns[0] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "false negative value should be meaningless",
                      UndefinedMetricWarning)
        fnr = np.repeat(np.nan, fns.shape)
    else:
        fnr = fns / fns[0]

    return fpr, tpr, tnr, fnr, thresholds


def calc_sens_spec(y_true, y_score, **kwargs):
    fpr, tpr, tnr, fnr, thresholds = any_curve(y_true, y_score)
    idx = 0
    maxv = -1e6
    for i, v in enumerate(tpr - fpr):
        if v > maxv:
            maxv = v
            idx = i
    #    idx = np.argmax(tpr - fpr)
    # print(tpr)
    # print(tnr)
    return tpr[idx], tnr[idx], thresholds[idx]


def analysis_pred_binary(y_true: Union[List, np.ndarray, pd.DataFrame], y_score: Union[List, np.ndarray, pd.DataFrame],
                         y_pred: Union[List, np.ndarray, pd.DataFrame] = None, alpha=0.95,
                         use_youden: bool = True, with_aux_ci: bool = False, reverse: bool = False):
    """

    Args:
        y_true:
        y_score:
        y_pred:
        alpha: 0.95
        use_youden: 是否使用youden指数
        with_aux_ci: 是否输出额外的CI
        reverse: bool，是否取反。

    Returns:

    """
    aux_ci = {}
    if isinstance(y_score, (list, tuple)):
        y_score = np.array(y_score)
    y_true = column_or_1d(np.array(y_true))
    assert sorted(np.unique(y_true)) == [0, 1], f"结果必须是2分类！"
    assert len(y_true) == len(y_score), '样本数必须相等！'
    if len(y_score.shape) == 2 and y_score.shape[1] == 2:
        y_score = column_or_1d(y_score[:, 1])
    elif len(y_score.shape) > 2:
        raise ValueError(f"y_score不支持>2列的数据！现在是{y_score.shape}")
    else:
        y_score = column_or_1d(y_score)
    if reverse:
        y_true = 1 - y_true
        y_score = 1 - y_score
    tpr, tnr, thres = calc_sens_spec(y_true, y_score)
    if y_pred is None:
        y_pred = np.array(y_score > (thres if use_youden else 0.5)).astype(int)
    acc = np.sum(y_true == y_pred) / len(y_true)
    tp = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    tn = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    # print(tp, tn, fp, fn)
    ppv = tp / (tp + fp + 1e-6)
    aux_ci['ppv'] = calc_value_95ci(tp, fp)
    npv = tn / (tn + fn + 1e-6)
    aux_ci['npv'] = calc_value_95ci(tn, fn)
    auc, ci = calc_95_CI(y_true, y_score, alpha=alpha, with_auc=True)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (fp + tn + 1e-6)
    aux_ci['sens'] = calc_value_95ci(tp, fn)
    aux_ci['spec'] = calc_value_95ci(tn, fp)
    f1 = 2 * tpr * ppv / (ppv + tpr)
    # print(tp, tn, fp, fn)
    if with_aux_ci:
        return acc, auc, map_ci(ci), tpr, map_ci(aux_ci['sens']), tnr, map_ci(aux_ci['spec']), \
            ppv, map_ci(aux_ci['ppv']), npv, map_ci(aux_ci['npv']), ppv, tpr, f1, thres
    else:
        return acc, auc, map_ci(ci), tpr, tnr, ppv, npv, ppv, tpr, f1, thres


def analysis_binary(y_true: Union[List, np.ndarray, pd.DataFrame],
                    y_pred: Union[List, np.ndarray, pd.DataFrame] = None,
                    with_aux_ci: bool = False, reverse: bool = False):
    """

    Args:
        y_true:
        y_score:
        y_pred:
        alpha: 0.95
        use_youden: 是否使用youden指数
        with_aux_ci: 是否输出额外的CI
        reverse: bool，是否取反。

    Returns:

    """
    aux_ci = {}
    y_true = column_or_1d(np.array(y_true))
    assert len(y_true) == len(y_pred), '样本数必须相等！'
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        y_score = column_or_1d(y_pred[:, 1])
    elif len(y_pred.shape) > 2:
        raise ValueError(f"y_score不支持>2列的数据！现在是{y_pred.shape}")
    else:
        y_pred = column_or_1d(y_pred)
    if reverse:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    acc = np.sum(y_true == y_pred) / len(y_true)
    tp = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    tn = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    # print(tp, tn, fp, fn)
    ppv = tp / (tp + fp + 1e-6)
    aux_ci['ppv'] = calc_value_95ci(tp, fp)
    npv = tn / (tn + fn + 1e-6)
    aux_ci['npv'] = calc_value_95ci(tn, fn)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (fp + tn + 1e-6)
    aux_ci['sens'] = calc_value_95ci(tp, fn)
    aux_ci['spec'] = calc_value_95ci(tn, fp)
    f1 = 2 * tpr * ppv / (ppv + tpr)
    # print(tp, tn, fp, fn)
    if with_aux_ci:
        return acc, tpr, map_ci(aux_ci['sens']), tnr, map_ci(aux_ci['spec']), \
            ppv, map_ci(aux_ci['ppv']), npv, map_ci(aux_ci['npv']), ppv, tpr, f1
    else:
        return acc, tpr, tnr, ppv, npv, ppv, tpr, f1


def IDI(pred_x, pred_y, gt, with_p: bool = False):
    """
    Calculate IDI metric.
    Args:
        gt: ground truth, group info
        pred_x: 旧模型预测结果
        pred_y: 新模型预测结果
        with_p: with p_value or not, default False

    Returns:

    """

    def _reshape(d_):
        return pd.DataFrame(np.reshape(np.array(d_), (-1, 1)))

    data = pd.concat([_reshape(gt), _reshape(pred_x), _reshape(pred_y)], axis=1)
    data.columns = ['gt', 'pred_x', 'pred_y']
    event = data[data['gt'] == 1]
    non_event = data[data['gt'] == 0]
    event_x_y = event['pred_x'] - event['pred_y']
    non_event_x_y = non_event['pred_x'] - non_event['pred_y']
    idi = np.mean(event_x_y) - np.mean(non_event_x_y)
    if with_p:
        return idi, idi / (((event_x_y.std() ** 2 + non_event_x_y.std() ** 2) ** 0.5) + 1e-6)
    else:
        return idi


def NRI(pred_x: Union[List, np.ndarray, pd.DataFrame], pred_y: Union[List, np.ndarray, pd.DataFrame],
        y_true: Union[List, np.ndarray, pd.DataFrame]):
    """
    计算NRI，为0时是最优状态，其他的都会或多或少有问题。

    Args:
        pred_x: 新模型预测结果
        pred_y: 旧模型预测结果
        y_true: 真实结果，观测的event状态

    Returns: NRI值。

    """
    y_true = column_or_1d(np.array(y_true, dtype=int))
    len_labels = len(np.unique(y_true))
    pred_x = column_or_1d(np.array(pred_x, dtype=int))
    pred_y = column_or_1d(np.array(pred_y, dtype=int))
    assert sorted(np.unique(y_true)) == [0, 1]
    event_num = np.sum(y_true)
    non_event_num = y_true.shape[0] - event_num
    matrix_event = sm.confusion_matrix(pred_x * y_true, pred_y * y_true)
    matrix_non_event = sm.confusion_matrix(pred_x * (1 - y_true), pred_y * (1 - y_true))
    tril = np.tril(np.ones((len_labels, len_labels)), -1)
    triu = np.triu(np.ones((len_labels, len_labels)), 1)
    a = (np.sum(matrix_event * tril) - np.sum(matrix_event * triu)) / event_num
    b = (np.sum(matrix_non_event * triu) - np.sum(matrix_non_event * tril)) / non_event_num
    return a + b


def calc_dice(p_cls, l_cls):
    # cal the inter & conv
    s = p_cls + l_cls
    inter = len(np.where(s >= 2)[0])
    conv = len(np.where(s >= 1)[0]) + inter
    try:
        dice = 2.0 * inter / conv
    except:
        print("conv is zeros when dice = 2.0 * inter / conv")
        dice = None
    return dice


def calc_iou(p_cls, l_cls):
    # cal the inter & conv
    s = p_cls + l_cls
    inter = len(np.where(s >= 2)[0])
    conv = len(np.where(s >= 1)[0])
    try:
        iou = inter / conv
    except:
        print("conv is zeros when dice = 2.0 * inter / conv")
        iou = None
    return iou


def calc_sa(p_cls, l_cls):
    # cal the inter & conv
    error = np.bitwise_xor(p_cls, l_cls) & l_cls
    try:
        sa = 1 - np.sum(error) / np.sum(l_cls)
    except:
        print("SA segmentation is error!")
        sa = None
    return sa


def calc_os(p_cls, l_cls):
    # cal the inter & conv
    error = np.bitwise_xor(p_cls, l_cls) & p_cls
    try:
        over_s = np.sum(error) / (np.sum(l_cls) + np.sum(p_cls))
    except:
        print("Over segmentation is error!")
        over_s = None
    return over_s


def calc_us(p_cls, l_cls):
    # cal the inter & conv
    error = np.bitwise_xor(p_cls & l_cls, l_cls)
    try:
        us = np.sum(error) / (np.sum(l_cls) + np.sum(np.bitwise_xor(p_cls, l_cls) & p_cls))
    except:
        print("Under segmentation is error!")
        us = None
    return us


def calc_asd(p_cls, l_cls):
    asd = compute_average_surface_distance(p_cls[np.newaxis, np.newaxis, :], l_cls[np.newaxis, np.newaxis, :])
    return float(asd)


def calc_hausdorff_distance(p_cls, l_cls):
    hd = compute_hausdorff_distance(p_cls[np.newaxis, np.newaxis, :], l_cls[np.newaxis, np.newaxis, :])
    return float(hd)


def seg_eval(pred, label, clss=[0, 1]):
    """
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    eval_matric = [None] * Ncls
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width], dtype=np.uint8)
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width], dtype=np.uint8)
        label_cls[np.where(label == cls)] = 1

        metric = [calc_dice(pred_cls, label_cls), calc_iou(pred_cls, label_cls),
                  calc_sa(pred_cls, label_cls), calc_os(pred_cls, label_cls), calc_us(pred_cls, label_cls),
                  calc_asd(pred_cls, label_cls), calc_hausdorff_distance(pred_cls, label_cls)]
        eval_matric[idx] = metric

    return eval_matric


def get_time_dependent_gt(survival: pd.DataFrame, time, id_col='ID', duration_col='duration', event_col='event'):
    """
    获取基于时间的Time-dependent label数据，基于incident/dynamic计算ground truth.

    Args:
        survival: 生存信息
        time: 计算时间依赖的数据截断时间
        id_col: ID列名
        duration_col: 时间列名
        event_col: 状态列名

    Returns:

    """
    sur = []
    for idx, row in survival.iterrows():
        if row[duration_col] > time:
            sur.append([row[id_col], 0])
        elif row[event_col] == 1:
            sur.append([row[id_col], 1])
    sur = pd.DataFrame(sur, columns=[id_col, 'label'])
    if sur.empty:
        print(f'随访时间太短，设置的随访时间{time}没有样本！')
    elif len(np.unique(sur['label'])) == 1:
        print(f"设置的随访时间{time}有问题！造成只有一种样本类型{np.unique(sur['label'])}")
    return sur


if __name__ == '__main__':
    y_true_ = [0, 0, 1, 1, 1, 1, 0]
    y_pred_ = [1, 1, 0, 0, 0, 0, 1]
    event_ = [1, 1, 0, 0, 0, 0, 1]
    y_pred_1 = [0.51, 0.61, 0.0, 0.01, 0.53, 0.99, 0.88]
    y_pred_2 = [1, 0.61, 1, 0.01, 0.53, 0.99, 0.88]
    print(analysis_pred_binary(y_true_, y_pred_1, with_aux_ci=True))
    print(calc_value_95ci(95, 98))
    print(calc_array_95ci(y_pred_))
    print(IDI(pred_x=y_pred_1, pred_y=y_pred_2, gt=y_true_))
    print(NRI(y_true_, event_, event_))
