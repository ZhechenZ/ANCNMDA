# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:25:44 2022

@author: zzc
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_disease, num_mirna, factor_num):
        super(GMF, self).__init__()

        self.num_disease = num_disease
        self.num_mirna = num_mirna
        self.predict_layer = nn.Linear(factor_num, 1)

        # self._init_weight_()

    # def _init_weight_(self):
    def forward(self, mirna, disease):
        output_GMF = disease * mirna
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1)


class MLP(nn.Module):
    def __init__(self, num_disease, num_mirna, factor_num, num_layers, dropout):
        super(MLP, self).__init__()

        self.num_disease = num_disease
        self.num_mirna = num_mirna
        self.embedding = nn.Linear(factor_num * 2, factor_num * (2 ** (num_layers - 0)))
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weight_()

    def _init_weight_(self):

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, mirna, disease):
        interaction = torch.cat((disease, mirna), -1)
        interaction = self.embedding(interaction)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.view(-1)


class NCF(nn.Module):
    def __init__(self, num_disease, num_mirna, factor_num, num_layers, num_heads, dropout):
        super(NCF, self).__init__()

        self.dropout = dropout
        # self.disease = disease
        # self.mirna = mirna
        self.num_layers = num_layers  # MLP的层数

        self.d_fc = nn.Linear(num_disease, factor_num)
        self.m_fc = nn.Linear(num_mirna, factor_num)
        self.embedding = nn.Linear(factor_num * 2, factor_num * (2 ** (num_layers - 0)))

        MLP_modules = []

        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())

        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        # 全连接层
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.predict_layer.weight, gain=gain)
        nn.init.xavier_normal_(self.embedding.weight, gain=gain)

        # MLP多层
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_grad()
        '''

    def forward(self, mirna, disease):

        output_GMF = mirna * disease  # 1 * 512
        interaction = torch.cat((mirna, disease), -1)  # 1 * 1024
        interaction = self.embedding(interaction)
        output_MLP = self.MLP_layers(interaction)  # 1 * 512
        # torch.cat拼接的操作
        # a==0.05, concat = torch.cat((0.05 * output_GMF, 0.95 * output_MLP),-1)
        concat = torch.cat((output_GMF, output_MLP), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)
