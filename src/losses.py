import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F


class QACrossEntropyLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', heads_reduction='mean'):
        assert heads_reduction in ['mean', 'sum']
        super().__init__(size_average, reduce, reduction)
        self.loss_fct = nn.CrossEntropyLoss()
        self.heads_reduction = heads_reduction

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        start_loss = self.loss_fct(start_logits, start_positions)
        end_loss = self.loss_fct(end_logits, end_positions)
        total_loss = start_loss + end_loss
        if self.heads_reduction == 'mean':
            total_loss = (start_loss + end_loss)/2
        return total_loss


class SmoothCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', smoothing=0.0, heads_reduction='mean'):
        assert heads_reduction in ['mean', 'sum']
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.heads_reduction = heads_reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward_head(self, inputs, targets):
        targets = self._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        start_loss = self.forward_head(start_logits, start_positions)
        end_loss = self.forward_head(end_logits, end_positions)
        total_loss = start_loss + end_loss

        if self.heads_reduction == 'mean':
            total_loss = (start_loss + end_loss)/2

        return total_loss


"""
class AggregatedLoss(nn.Module)
    __constants__ = ['reduction']

    def __init__(self, loss, size_average=None, reduce=None,
                 reduction='mean', heads_reduction='mean'):
        assert heads_reduction in ['mean', 'sum']
        super().__init__(size_average, reduce, reduction)
        self.loss_fct = nn.CrossEntropyLoss()
        self.heads_reduction = heads_reduction

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        start_loss = self.loss_fct(start_logits, start_positions)
        end_loss = self.loss_fct(end_logits, end_positions)
        total_loss = start_loss + end_loss
        if self.heads_reduction == 'mean':
            total_loss = (start_loss + end_loss)/2
        return total_loss
    def __init__(self, loss_fn, heads_reduction='mean'):
        assert heads_reduction in ['mean', 'sum']
        super().__init__(size_average, reduce, reduction)
        self.loss_fct = nn.CrossEntropyLoss()
        self.heads_reduction = heads_reduction
"""
