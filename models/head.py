import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

N_TARGETS = 30
N_Q_TARGETS = 21
N_A_TARGETS = 9

class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

def lin_layer(n_in, n_out, dropout):
    return nn.Sequential(nn.Linear(n_in, n_out), GELU(), nn.Dropout(dropout))
    
class MultiSampleClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.num_samples = num_samples
        for i in range(num_samples):
            setattr(self, 'dropout{}'.format(i), nn.Dropout(dropout_rate))
            setattr(self, 'fc{}'.format(i), nn.Linear(in_features, out_features))

    def forward(self, x):
        logits = []
        for i in range(self.num_samples):
            dropout = getattr(self, 'dropout{}'.format(i))
            fc = getattr(self, 'fc{}'.format(i))
            x_ = dropout(x)
            x_ = fc(x_)
            logits.append(x_)
        return torch.stack(logits).mean(dim=0) 

class Head(nn.Module):
    def __init__(self, n_h=256, n_feats=0, n_bert=768, dropout=0.2):
        super().__init__()
        n_x = n_feats + 2 * n_bert
        self.lin = lin_layer(n_in=n_x, n_out=n_h, dropout=dropout)
        self.lin_q = lin_layer(n_in=n_feats + n_bert, n_out=n_h, dropout=dropout)
        self.lin_a = lin_layer(n_in=n_feats + n_bert, n_out=n_h, dropout=dropout)
        self.head_q = MultiSampleClassifier(2 * n_h, N_Q_TARGETS)
        self.head_a = MultiSampleClassifier(2 * n_h, N_A_TARGETS)

    def forward(self, x_q_bert, x_a_bert):
        # x_q = self.lin_q(torch.cat([x_feats, x_q_bert], dim=1))
        # x_a = self.lin_a(torch.cat([x_feats, x_a_bert], dim=1))
        #x = self.lin(torch.cat([x_feats, x_q_bert, x_a_bert], dim=1))
        x_q = self.lin_q(x_q_bert)
        x_a = self.lin_a(x_a_bert)
        x = self.lin(torch.cat([x_q_bert, x_a_bert], dim=1))
        x_q = self.head_q(torch.cat([x, x_q], dim=1))
        x_a = self.head_a(torch.cat([x, x_a], dim=1))
        return torch.cat([x_q, x_a], dim=1)
