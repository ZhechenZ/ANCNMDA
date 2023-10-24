# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:39:26 2022

@author: zzc
"""
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
from load_data import build_graph
from model import ANCNMDA
from sklearn.model_selection import KFold
from sklearn import metrics
import time
from case_study import Breast_Cancer, Lung_Cancer, Esophageal_Neoplasms, Pancreatic_Neoplasms
import warnings
import dgl


# #####################   TRAIN   ########################
# 模型架构,实验数据,损失函数
def Train(path, epochs, attn_heads, attn_size, out_dim, dropout, MLP_num_layers,
          slope, lr, wd, random_seed, model):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device('cuda:1' if torch.cuda.is_available() is True else 'cpu')
    ID, IM, samples, g, disease_vertices, mirna_vertices = build_graph(path, random_seed)
    # study_samples = Pancreatic_Neoplasms()


    g = g.to(device)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    auc_result = []
    acc_result = []
    pre_result = []  # p = tp/(tp+fp)
    recall_result = []  # r = tp/(tp+fn)
    f1_result = []  # f = 2/(1/p+1/r)
    prc_result = []
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    #  五折交叉验证
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in kf.split(samples[:, 2]):  # 取列表中所有行的第三列
        # print(train_idx,'string',test_idx) 将10860个索引分成五份，train_idx占了4/5，test_idx占了1/5
        i += 1
        print('##########   Training  For  Fold  {}   ###########'.format(i))
        # 将训练集的索引设置为1，其他的索引设置为0
        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1
        # s10860 * 4,多出来的一列为train, train列为1的用来训练,0用来测试
        # print('## train data: {} , test data: {}'.format(samples_df['train'].value_counts()))
        train_samples = samples_df.loc[samples_df['train'] == 1]
        # print(train_samples.shape[0]) 8688
        test_samples = samples_df.loc[samples_df['train'] == 0]


        train_samples = train_samples.values.tolist()
        test_samples = test_samples.values.tolist()
        train_samples = torch.tensor(train_samples, dtype=torch.long)
        test_samples = torch.tensor(test_samples, dtype=torch.long)
        # study_samples = torch.tensor(study_samples, dtype = torch.long)

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))
        edge_data = {'train': train_tensor}

        # 对图的训练边进行标记
        edge_data = {key: edge_data[key].cuda() for key in edge_data}
        disease_vertices = torch.tensor(disease_vertices)
        mirna_vertices = torch.tensor(mirna_vertices)
        disease_vertices = disease_vertices.to(device)
        mirna_vertices = mirna_vertices.to(device)

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)  # g_train是根据训练数据集构造的一个子图
        g_train = g_train.to(device)

        # 训练标签和测试标签用于最后的对比
        # label_train = g_train.edata['label'].unsqueeze(1)
        label_train = train_samples[:, 2]
        # src_train, dst_train = g_train.all_edges()
        label_train = torch.tensor(label_train, dtype=torch.float)
        label_train = label_train.to(device)
        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        # src_test, dst_test = g.find_edges(test_eid)
        label_test = test_samples[:, 2]
        label_test = torch.tensor(label_test, dtype=torch.float)
        label_test = label_test.to(device)
        g_test = g.edge_subgraph(test_eid, preserve_nodes=True)
        g_test = g_test.to(device)
        print('### Training Samples :', len(train_samples[:, 0]))
        print('### Testing Samples :', len(test_samples[:, 0]))
        if model == 'ANCNMDA':
            model = ANCNMDA(
                G=g,
                feature_attn_size=attn_size,
                num_layers=MLP_num_layers,
                num_heads=attn_heads,
                num_mirnas=IM.shape[0],
                num_diseases=ID.shape[0],
                d_sim_dim=ID.shape[1],
                m_sim_dim=IM.shape[1],
                out_dim=out_dim,
                dropout=dropout,
                slope=slope
            )
        model = model.to(device)
        loss_function = nn.BCELoss()
        loss_function = loss_function.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                # prediction_train = model(g_train, src_train, dst_train, train_samples)
                prediction_train = model(g_train, train_samples)
                prediction_train.to(device)
                loss_train = loss_function(prediction_train, label_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                # prediction_test = model(g_test, src_test, dst_test, test_samples)
                prediction_test = model(g_test, test_samples)
                prediction_test = prediction_test.to(device)
                loss_test = loss_function(prediction_test, label_test)

            prediction_train_ = np.squeeze(prediction_train.cpu().detach().numpy())
            prediction_test_ = np.squeeze(prediction_test.cpu().detach().numpy())

            label_train_ = np.squeeze(label_train.cpu().detach().numpy())
            label_test_ = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_, prediction_train_)
            test_auc = metrics.roc_auc_score(label_test_, prediction_test_)

            prediction_value = [0 if i < 0.5 else 1 for i in prediction_test_]  # 计算测试集的联系
            acc_value = metrics.accuracy_score(label_test_, prediction_value)
            precisions_value = metrics.precision_score(label_test_, prediction_value)
            recall_value = metrics.recall_score(label_test_, prediction_value)
            f1_value = metrics.f1_score(label_test_, prediction_value)
            end_time = time.time()


            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Test Loss: %.4f' % loss_test.cpu().detach().numpy(),
                      'Accuracy: %.4f' % acc_value, 'Precision: %.4f' % precisions_value, 'Recall: %.4f' % recall_value,
                      'F1-score: %.4f' % f1_value, 'Train Auc: %.4f' % train_auc, 'Test Auc: %.4f' % test_auc, 'Time: %.2f' % (end_time - start_time)
                      )

        model.eval()
        # 每一折训练完之后, 将测试集放入模型中进行测试
        with torch.no_grad():
            score_test = model(g_test, test_samples)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        prediction_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, prediction_test)
        pre_test = metrics.precision_score(label_test_cpu, prediction_test)
        recall_test = metrics.recall_score(label_test_cpu, prediction_test)
        f1_test = metrics.f1_score(label_test_cpu, prediction_test)

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test pre: % .4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1-score: %.4f' % f1_test,
              'Test Prc: %.4f' % test_prc, 'Test Auc: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('##########  Training Finished  ##########')
    '''
    case_study_result = model(g, study_samples)
    # print(case_study_result.cpu().tolist())
    case_study_result = case_study_result.cpu().tolist()
    data = {'prediction_score1-2': case_study_result}
    df = pd.DataFrame(data)
    df.to_csv('case_study_result/Pancreatic_Cancer.csv', index = False)
    '''
    print('--------------------------------------------------------------------------------')
    print('--Auc mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          '--Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          '--Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          '--Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          '--F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          '--PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result))
          )

    return fprs, tprs, auc_result, precisions, recalls, prc_result



