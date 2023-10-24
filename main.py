# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:25:08 2022

@author: zzc
"""

import warnings
from train import Train
from draw_pic import plot_prc_curves, plot_auc_curves
warnings.filterwarnings('ignore')

# print('='*30, 'Only GMF', '='*30)

for i in range(10):
    print('---------------------------------------------------------------------------------------------- No. ', i + 1, ' Training ----------------------------------------------------------------------------------------------')
    '''
    fprs = []
    tprs = []
    auc = []
    precisions = []
    recalls = []
    prc = []
    '''
    fprs, tprs, auc, precisions, recalls, prc = Train(path = "data",
                                                      epochs = 500,
                                                      attn_size = 64,
                                                      attn_heads = 6,
                                                      out_dim = 64,
                                                      dropout = 0.6,
                                                      slope = 0.2,
                                                      lr = 0.001,
                                                      wd = 5e-3,
                                                      random_seed = 1234,
                                                      model = 'ANCNMDA',
                                                      MLP_num_layers = 3
                                                      )
    plot_auc_curves(fprs, tprs, auc, i+1, 'auc')
    plot_prc_curves(precisions, recalls, prc, i+1, 'prc')




