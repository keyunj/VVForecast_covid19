import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["NLLLoss", "BalanceNLLLoss", "DiceLoss", "HybridLoss"]


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, input, target):
        input_log_sm = -F.log_softmax(input, 1)
        return torch.mean(input_log_sm[:, 0] * (1 - target) + input_log_sm[:, 1] * target)


class BalanceNLLLoss(nn.Module):
    def __init__(self):
        super(BalanceNLLLoss, self).__init__()

    def forward(self, input, target):
        input_log_sm = -F.log_softmax(input, 1)
        loss_pos = torch.sum(input_log_sm[:, 1] * target)
        N = torch.sum(target)
        log_neg = torch.flatten(input_log_sm[:, 0] * (1 - target))
        _, neg_indices = torch.topk(log_neg, N)
        loss_neg = torch.sum(log_neg[neg_indices])
        return (loss_pos + loss_neg) / (2 * N) + F.cross_entropy(input, target)


def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    tranposed = tensor.permute(axis_order)
    return tranposed.contiguous().view(C, -1)


class TanimotoLoss(nn.Module):
    def __init__(self, smooth=1):
        super(TanimotoLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input_sm = F.softmax(input, 1)
        gt_onehot = torch.zeros_like(input_sm).scatter_(1, target[:, None].long(), 1)

        input_flat = flatten(input_sm)
        gt_flat = flatten(gt_onehot).float()

        intersect = (input_flat * gt_flat).sum(-1)
        input_sqr = (input_flat ** 2).sum(-1)
        gt_sqr = (gt_flat ** 2).sum(-1)

        tanimoto = (intersect + self.smooth) / (input_sqr + gt_sqr - intersect + self.smooth)
        return torch.mean(1.0 - tanimoto)


class HybridLoss(nn.Module):
    def __init__(self, losses, smooth=1, gamma=2, local_size=None, logits=True, k_weight=None):
        super(HybridLoss, self).__init__()
        self.k_weight = k_weight
        self.loss_dice = DiceLoss(smooth=smooth, logits=logits) if "_dice_" in losses else None
        self.loss_ce = nn.NLLLoss(weight=k_weight) if "_ce_" in losses else None
        self.loss_focal = FocalLoss(gamma=gamma, k_weight=k_weight) if "_focal_" in losses else None
        self.loss_mce = (
            MultiDiceLoss(k_weight=k_weight, smooth=smooth, logits=logits) if "_mce_" in losses else None
        )
        self.loss_local = (
            LocalMatchLoss(local_size=local_size, k_weight=k_weight) if "_local_" in losses else None
        )

    def forward(self, input, target):
        loss = 0
        if self.loss_dice is not None:
            loss += 20 * self.loss_dice(input, target.long())
        if self.loss_ce is not None:
            loss += self.loss_ce(torch.log(input), target.long())
            # loss += self.loss_ce(torch.log(input), target.long())
        if self.loss_focal is not None:
            loss += self.loss_focal(input, target.long())
        if self.loss_mce is not None:
            loss += 20 * self.loss_mce(input, target.long())
        if self.loss_local is not None:
            loss += self.loss_local(input, target.long())
        return loss


class MultiDiceLoss(nn.Module):
    def __init__(self, k_weight=None, smooth=1, logits=True):
        super(MultiDiceLoss, self).__init__()
        self.k_weight = k_weight
        self.smooth = smooth
        self.logits = logits

    def forward(self, input, target):
        if self.logits:
            input_sm = torch.softmax(input, dim=1)
        else:
            input_sm = input

        num_classes = input_sm.size(1)
        num_t = input_sm.size(2)
        loss = 0
        for k in range(num_classes):
            for t in range(num_t):
                input_k = input_sm[:, k, t]
                target_k = (target[:, t] == k).float()
                dice = (2 * (input_k * target_k).sum() + self.smooth) / (
                    (input_k ** 2).sum() + (target_k ** 2).sum() + self.smooth
                )
                loss += self.k_weight[k] * (1.0 - dice)

                input_k = 1 - input_k
                target_k = 1 - target_k
                dice = (2 * (input_k * target_k).sum() + self.smooth) / (
                    (input_k ** 2).sum() + (target_k ** 2).sum() + self.smooth
                )
                loss += self.k_weight[k] * (1.0 - dice)

        return loss / (num_classes * num_t * 2)


class LocalMatchLoss(nn.Module):
    def __init__(self, local_size=5, k_weight=None):
        super(LocalMatchLoss, self).__init__()
        self.local_size = local_size
        self.k_weight = k_weight

    def forward(self, input, target):
        b, k, t, h, w = input.size()
        gt_onehot = torch.zeros_like(input).scatter_(1, target[:, None], 1)
        input_smooth = F.adaptive_avg_pool2d(input.view(b, -1, h, w), output_size=self.local_size)
        gt_smooth = F.adaptive_avg_pool2d(gt_onehot.view(b, -1, h, w), output_size=self.local_size)
        loss = torch.mean(torch.abs(input_smooth - gt_smooth), dim=[0, 2, 3]).view(k, t)
        if self.k_weight is not None:
            loss *= self.k_weight[:, None]
        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, k_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.k_weight = k_weight

    def forward(self, input, target):
        b = input.size(0)

        k = len(self.k_weight)
        t = input.size(1)

        log_pt = torch.log(input)
        log_pt = log_pt.gather(1, target[:, None]).view(b, 1, t, -1)
        pt = input.gather(1, target[:, None]).view(b, 1, t, -1)

        if self.k_weight is not None:
            alpha_t = self.k_weight.gather(0, target.view(-1)).view(b, 1, t, -1)
            log_pt = log_pt * alpha_t

        loss = -((1 - pt) ** self.gamma) * log_pt
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.logits = logits

    def forward(self, input, target):
        if self.logits:
            input_sm = F.softmax(input, 1)
        else:
            input_sm = input
        gt_onehot = torch.zeros_like(input_sm).scatter_(1, target[:, None], 1)

        input_flat = flatten(input_sm)
        gt_flat = flatten(gt_onehot).float()

        intersect = (input_flat * gt_flat).sum(-1)
        input_sqr = (input_flat ** 2).sum(-1)
        gt_sqr = (gt_flat ** 2).sum(-1)

        dsc = (2 * intersect + self.smooth) / (input_sqr + gt_sqr + self.smooth)
        return 1.0 - dsc.mean()
