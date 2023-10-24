# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:08:07 2022

@author: zzc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, G, feature_attn_size, dropout, slope):  # dropout参数为了防止过拟合而设定

        super(Layer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)  # list->[0->382]
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)  # list->[383->877]

        self.G = G

        # 'type':878列 0-382为1,else为0    'm_sim':878*495,下面是0    'd_sim':878*383,上面是0
        # print(self.G.edges())    # nodes(): tensor([0->877])     edges(): tensor([0...687],[383...337])
        # print(torch.sum(self.G.ndata['type'],dim=0))  结果为383
        self.slope = slope
        self.m_f = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size,
                             bias=False)  # 权重矩阵 列数不变,行数变为feature_attn_size
        self.d_f = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    # 使用激活函数ReLu对权重进行重新设置
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_f.weight, gain=gain)
        nn.init.xavier_normal_(self.d_f.weight, gain=gain)

    def edge_attention(self, edges):  # 计算边的注意力euv
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(
            1)  # torch.sum对输入的数据求某一维度的和,dim=0竖着求和,dim=1横着求和
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):  # 对边的注意力系数进行归一化再求和
        # alpha注意力系数
        # alpha = 0.5
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, G):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_f(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_f(nodes.data['m_sim']))}, self.mirna_nodes)

        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)

        return self.G.ndata.pop('h')


# 多头注意里机制
class MultiHeadLayer(nn.Module):
    def __init__(self, G, num_heads, dropout, slope, feature_attn_size, merge='cat'):
        super(MultiHeadLayer, self).__init__()

        self.G = G
        self.dropout = dropout
        self.slope = slope  # 激活函数的斜率
        self.merge = merge  # 拼接方式
        self.num_heads = num_heads
        self.feature_attn_size = feature_attn_size
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(Layer(G, feature_attn_size, dropout, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)  # dim=1,左右拼接
        else:
            return torch.mean(torch.stack(head_outs), dim=0)  # dim=0,上下拼接



