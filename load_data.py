# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:25:44 2022

@author: zzc
"""

import numpy as np
import pandas as pd
import dgl
import torch


def load_data(path, random_seed):
    D_GSM = np.loadtxt(path + '/D_GSM.txt')  # 疾病高斯矩阵
    D_SSM1 = np.loadtxt(path + '/D_SSM1.txt')  # 疾病语义相似矩阵1
    D_SSM2 = np.loadtxt(path + '/D_SSM2.txt')  # 疾病语义相似矩阵2
    M_FSM = np.loadtxt(path + '/M_FSM.txt')  # miRNA功能相似矩阵
    M_GSM = np.loadtxt(path + '/M_GSM.txt')  # miRNA高斯相似矩阵
    all_associations = pd.read_csv(path + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])  # 已知所有联系
    D_SSM = (D_SSM1 + D_SSM2) / 2

    ID = D_SSM
    IM = M_FSM

    # 构造ID,IM矩阵
    for i in range(ID.shape[0]):
        for j in range(ID.shape[0]):
            if ID[i, j] == 0:
                ID[i, j] = D_GSM[i][j]

    for i in range(IM.shape[0]):
        for j in range(IM.shape[0]):
            if IM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]


    # 筛选miRNA-disease正样本和与正样本数相同的负样本
    known_associations = all_associations.loc[all_associations['label'] == 1]  # 将所有样本均设为有联系的样本
    unknown_associations = all_associations.loc[all_associations['label'] == 0]  # 将所有样本均设为无联系的样本
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed,
                                                  axis=0)  # 从这个没有关系的样本中随机选5430个作为负样本
    sample_df = known_associations.append(random_negative)
    # 指针重置
    sample_df.reset_index(drop=True, inplace=True)
    samples = sample_df.values  # 列表的形式,获得新的样本编号


    return ID, IM, samples


def build_graph(path, random_seed):
    ID, IM, samples = load_data(path, random_seed)
    # 构建miRNA-diseases二元异构图
    g = dgl.DGLGraph()  # 构建一个没有边和节点的图
    g.add_nodes(ID.shape[0] + IM.shape[0])  # 添加n+m个节点
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    # 0-382设为疾病节点，并传入特征
    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    # 383-877设为miRNA节点，并传入特征
    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[0])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    # 让指针从0开始，原本节点标签从1开始
    disease_ids = list(range(1, ID.shape[0] + 1))  # 列表{1->383}
    mirna_ids = list(range(1, IM.shape[0] + 1))  # 列表{1->495}

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]
    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    return ID, IM, samples, g, sample_disease_vertices, sample_mirna_vertices

