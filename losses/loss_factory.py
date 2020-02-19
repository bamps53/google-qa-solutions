import sys

sys.path.insert(0, '../..')
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from . import functions

#from https://github.com/qubvel/segmentation_models.pytorch
class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class CEDiceLoss(DiceLoss):
    __name__ = 'ce_dice_loss'

    def __init__(self, eps=1e-7, activation='softmax2d'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        y_pr = torch.nn.Softmax2d()(y_pr)
        ce = self.bce(y_pr, y_gt)
        return dice + ce


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


class WeightedBCELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32).cuda()

    def forward(self, logit, truth):
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        loss = (loss * self.weight).mean()
        return loss

class MarginRankLoss(nn.Module):
    def __init__(self, weight, margin=0.1):
        super().__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32).cuda()
        self.margin = margin

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        if batch_size % 2 == 0:
            outputs1, outputs2 = outputs.sigmoid().contiguous().view(2, batch_size // 2, outputs.size(-1))
            targets1, targets2 = targets.contiguous().view(2, batch_size // 2, outputs.size(-1))
            # 1 if first ones are larger, -1 if second ones are larger, and 0 if equals.
            ordering = (targets1 > targets2).float() - (targets1 < targets2).float()
            margin_rank_loss = (-ordering * (outputs1 - outputs2) + self.margin).clamp(min=0.0)
            margin_rank_loss = (margin_rank_loss * self.weight).mean()
        else:
            margin_rank_loss = 0
        return margin_rank_loss

class WeightedBCEMarginRankLoss(nn.Module):
    def __init__(self, weight, margin=0.1, alpha=0.5):
        super().__init__()
        self.bce = WeightedBCELoss(weight)
        self.margin = MarginRankLoss(weight, margin=margin)
        self.alpha = alpha

    def forward(self, outputs, targets):
        bce = self.bce(outputs, targets)
        margin = self.margin(outputs, targets)

        return bce * self.alpha + margin * (1 - self.alpha)

def get_loss(config):
    if config.loss.name == 'BCEDice':
        criterion = BCEDiceLoss(eps=1.)
    elif config.loss.name == 'CEDice':
        criterion = CEDiceLoss(eps=1.)
    elif config.loss.name == 'WeightedBCE':
        criterion = WeightedBCELoss(weight=np.load('data/label_weights.npy'))
    elif config.loss.name == 'WeightedBCEMargin':
        criterion = WeightedBCEMarginRankLoss(weight=np.load('data/label_weights.npy'))
    elif config.loss.name == 'BCE':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif config.loss.name == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif config.loss.name == 'CE':
        #criterion = nn.CrossEntropyLoss(reduction='mean')
        criterion = CELoss()
    else:
        raise Exception('Your loss name is not implemented. Please choose from [BCEDice, CEDice, WeightedBCE, BCE]')
    return criterion
