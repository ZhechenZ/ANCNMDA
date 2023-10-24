# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:58:12 2022

+
@author: 13734
"""


import torch.nn as nn
import torch
from layers import MultiHeadLayer
from ncf import NCF, GMF, MLP
import torch.nn.functional as F



class GATNCF(nn.Module):
    def __init__(self, G, feature_attn_size, num_layers, num_heads, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope):
        super(GATNCF, self).__init__()
        self.G = G
        self.feature_attn_size = feature_attn_size
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.d_sim_dim = d_sim_dim
        self.m_sim_dim = m_sim_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.slope = slope
        self.gat = MultiHeadLayer(G, num_heads, dropout, slope, feature_attn_size)
        self.ncf = NCF(num_diseases, num_mirnas, out_dim, num_layers, num_heads, dropout)
        self.metapath_layers = nn.ModuleList()
        self.m_f = nn.Linear(feature_attn_size * num_heads, out_dim)
        self.d_f = nn.Linear(feature_attn_size * num_heads, out_dim)
        self.gmf = GMF(num_diseases, num_mirnas, factor_num = out_dim)
        self.mlp = MLP(num_diseases, num_mirnas, out_dim, num_layers, dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_f.weight, gain=gain)
        nn.init.xavier_normal_(self.d_f.weight, gain=gain)

    def forward(self, G, samples):
        h_agg = self.gat(G)

        disease_0 = h_agg[:self.num_diseases]
        mirna_0 = h_agg[self.num_diseases:self.num_diseases+self.num_mirnas]


        disease_1 = self.d_f(disease_0)  # 383 * 64
        mirna_1 = self.m_f(mirna_0)  # 495 * 64

        return F.sigmoid(self.ncf(mirna_1[samples[:, 0] - 1], disease_1[samples[:, 1] - 1]))