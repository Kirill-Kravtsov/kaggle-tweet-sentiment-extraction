import torch.nn as nn


class QACrossEntropyLoss(nn.modules.loss._Loss):
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
