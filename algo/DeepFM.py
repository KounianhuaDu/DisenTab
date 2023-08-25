import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from regularizations.InfoMinMax import INFOMIN


class FMLayer(nn.Module):
    def __init__(self):
        super(FMLayer, self).__init__()
             
    def forward(self, x):
        # x: B*F*E
        sum_square = (x.sum(1)) * (x.sum(1))
        square_sum = (x * x).sum(1)
        inter = sum_square-square_sum
        inter = inter/2
        return inter.sum(1,keepdim=True)

class Linear(nn.Module):
    def __init__(self, num_feat, padding_idx, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(num_feat+1, output_dim, padding_idx = padding_idx)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias

class DeepFM(nn.Module):
    def __init__(self, num_feat, num_fields, padding_idx, embedding_size, dropout_prob, args, valid_feat=None):
        super(DeepFM, self).__init__()
        
        self.w = nn.Embedding(num_feat+1, embedding_size, padding_idx = padding_idx)
        self.linear = Linear(num_feat, padding_idx)
        nn.init.xavier_uniform_(self.w.weight.data)
        self.fm = FMLayer()
        self.deep = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(num_fields*embedding_size, 128),
                nn.ReLU(),
                nn.LayerNorm(128, elementwise_affine=False, eps=1e-8),
                nn.Dropout(p=dropout_prob),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64, elementwise_affine=False, eps=1e-8),
                nn.Linear(64, 1)
            )
        self.infomin = INFOMIN(embedding_size, 1)
    
    def feature_dis_loss(self, x):
        x = self.w(x)
        # x: [bs, fields, embedding_size]
        B = x.shape[0]
        F = x.shape[1]
        feature_batch = torch.repeat_interleave(torch.arange(B), F).to(x.device)
        x = x.reshape(-1, x.shape[-1])
        feat_dis_loss = self.infomin(x, feature_batch)
        return feat_dis_loss
    
    def forward(self, x_user, x_item, x_context, user_hist, hist_len):
        X = torch.cat((x_user, x_item, x_context), dim=1)
        x_emb = self.w(X)
        dnn_out = self.deep(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm(x_emb)
        # DNN + fm second order + fm first order
        output = dnn_out + fm_out + self.linear(X)
        return torch.sigmoid(output.squeeze(1))

