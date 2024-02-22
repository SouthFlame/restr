import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def loss_calc(pred, label, rank):
    label = Variable(label.long()).to(rank)
    criterion = CrossEntropy2d()
    return criterion(pred, label)

def lr_poly(base_lr, end_lr, iter, max_iter, power, i_warm):
    if iter < i_warm:
        coef = iter / i_warm
        # coef *= (1 - i_warm/max_iter) ** power
    else:
        coef = (1 - (iter-i_warm)/(max_iter-i_warm)) ** power
    return (base_lr-end_lr) * coef + end_lr

def lr_cos(base_lr, end_lr, iter, max_iter, i_warm):
    if iter < i_warm:
        coef = iter+1 / i_warm
        return base_lr * coef
    else:
        coef = 0.5*(1 + math.cos((iter-i_warm)/(max_iter-i_warm)*math.pi))
        return (base_lr-end_lr) * coef + end_lr

def adjust_learning_rate(optimizer, i_iter, lr, end_lr, num_steps, power, i_warm = 0.0, is_cos=False):
    lr = lr_poly(lr, end_lr, i_iter, num_steps, power, i_warm) if is_cos is False else lr_cos(lr, end_lr, i_iter, num_steps, i_warm)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 0.1

    return lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging"""
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)