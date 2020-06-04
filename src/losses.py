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

    def __init__(self, weight=None, reduction='mean', smoothing=0.2, heads_reduction='mean'):
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



class SoftCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', heads_reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.heads_reduction = heads_reduction

    @staticmethod
    def _pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
        # neg_weight for when pred position < target position
        # pos_weight for when pred position > target position
        gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
        gap = gap.type(torch.float32)
        return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        loss_fct = nn.CrossEntropyLoss(reduce='none') # do reduction later
        
        start_pos = self._pos_weight(start_logits, start_positions, 1, 1)
        end_pos = self._pos_weight(end_logits, end_positions, 1, 1)

        start_loss = loss_fct(start_logits, start_positions) * start_pos
        end_loss = loss_fct(end_logits, end_positions) * end_pos
        
        start_loss = torch.mean(start_loss)
        end_loss = torch.mean(end_loss)
        total_loss = start_loss + end_loss

        if self.heads_reduction == 'mean':
            total_loss = (start_loss + end_loss)/2
        return total_loss


class JaccardApproxLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, logits, targets):
        #t = torch.Tensor([0.5]).to(logits.device)
        #logits = (logits >= t).long().squeeze(-1)

        #logits = torch.relu(torch.sign(logits)).squeeze(-1)
        #print(pred_mask.requires_grad)
        #print(pred_mask.shape, targets.shape)
        logits = logits.squeeze(-1)

        bce = torch.nn.BCEWithLogitsLoss()(logits, targets.float())

        logits = torch.sigmoid(logits)
        intersection = logits * targets
        num = intersection.sum(dim=1)
        denum = targets.sum(dim=1) + logits.sum(dim=1) - num
        jaccard = (-num / denum).float().mean() * 10

        return (jaccard + bce)/2
 
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


class JaccardLstm(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.lstm1 = nn.GRU(5, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(256 + 5, 256, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(1024 + (256 + 5)*2, 1024)
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, start_logits, end_logits, bin_sentiment, new_words, bin_sentiment_words):
        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        x = torch.stack((
                start_probs,
                end_probs,
                bin_sentiment,
                new_words,
                bin_sentiment_words),
            dim=-1)

        h_lstm1, _ = self.lstm1(x)
        x = torch.cat((h_lstm1, x), dim=-1)
        h_lstm2, _ = self.lstm2(x)

        prev_avg_pool = torch.mean(x, dim=1)
        prev_max_pool, _ = torch.max(x, dim=1)

        avg_pool = torch.mean(h_lstm2, dim=1)
        max_pool, _ = torch.max(h_lstm2, dim=1)

        h_conc = torch.cat((max_pool, avg_pool, prev_avg_pool, prev_max_pool), dim=1)
        x = self.linear1(h_conc)
        x = self.linear2(x).view(-1)
        return x


def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())


class JaccardNNApproxLoss(_WeightedLoss):

    def __init__(self, path_weights, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.model = JaccardLstm()
        self.model.load_state_dict(torch.load(path_weights))
        #for p,data in self.model.named_parameters():
        #    print(p)
            #p.register_backward_hook(printgradnorm)
        #self.model.lstm1.register_backward_hook(printgradnorm)
        #print(self.model.lstm1.weight_ih_l0.norm())
        self.model.cuda()
        self.default_loss = QACrossEntropyLoss()

    def forward(self, start_logits, end_logits,
                start_positions, end_positions,
                bin_sentiment, new_words, bin_sentiment_words):
        self.model.zero_grad()
        #print(start_logits.requires_grad)
        #for p in self.model.parameters():
        #    p.grad.data.zero_()
        #print(self.model.lstm1.weight_ih_l0.norm())

        batch_jaccard_pred = self.model(
            start_logits, end_logits, bin_sentiment,
            new_words, bin_sentiment_words)
        #print(-torch.mean(batch_jaccard_pred))

        default_loss = self.default_loss(start_logits, end_logits, start_positions, end_positions)
        return 1.5*torch.mean(1 - torch.sigmoid(batch_jaccard_pred)) + default_loss
