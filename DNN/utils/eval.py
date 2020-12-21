from __future__ import print_function, absolute_import
import torch
import numpy as np

__all__ = ["angledifference", "accuracy", "cal_dice", "cal_all_metric"]


def angledifference(cosine_similarity):
    if torch.is_tensor(cosine_similarity):
        angdiff = torch.acos(cosine_similarity.clamp(-1, 1))
    else:
        angdiff = np.arccos(cosine_similarity)
    return angdiff.mean()


def cal_dice(input, target):
    smooth = 1e-3
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
    else:
        input_var = input.astype(np.float32)
    input_cal = 1.0 * (input_var > 0.5)
    target_cal = 1.0 * (target > 0.5)
    intersection = input_cal * target_cal
    dice = (2 * intersection.sum() + smooth) / (input_cal.sum() + target_cal.sum() + smooth)
    return dice


def cal_all_metric(input, target):
    """
    calculate all metric used in segmentation task
    """
    smooth = 1
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
    else:
        input_var = input.astype(np.float32)
    intersection_sum = (input_var * target).sum()
    input_sum = input_var.sum()
    target_sum = target.sum()
    dice = (2 * intersection_sum + smooth) / (input_sum + target_sum + smooth)
    precision = (intersection_sum + smooth) / (input_sum + smooth)
    recall = (intersection_sum + smooth) / (target_sum + smooth)
    return (dice, precision, recall)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
