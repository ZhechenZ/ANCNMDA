import pandas as pd
import numpy as np


def Breast_Cancer():
    # 用到miRNA-disease关联2(pairs2)的数据
    # disease number:50
    all_associations = pd.read_csv('data/all_mirna_disease_pairs.csv', names = ['miRNA', 'disease', 'label'])
    all_associations = all_associations.values.tolist()
    '''
    new_associations = []
    for i in all_associations:
        if i[1] != 50:
            new_associations.append(i)
    miRNA = []
    disease = []
    label = []
    for i in new_associations:
        miRNA.append(i[0])
        disease.append(i[1])
        label.append(i[2])
    '''
    for i in all_associations:
        if i[1] == 50 and i[2] == 1:
            i[2] = 0

    miRNA = []
    disease = []
    label = []
    for i in all_associations:
        miRNA.append(i[0])
        disease.append(i[1])
        label.append(i[2])
    data = {'miRNA': miRNA, 'disease': disease, 'label': label}
    df = pd.DataFrame(data)
    df.to_csv('data/all_mirna_disease_pairs1-2.csv', index = False)
    # data = {'miRNA': miRNA, 'disease': disease, 'label': label}
    # df = pd.DataFrame(data)
    # df.to_csv('data/all_mirna_disease_pairs1.csv', index = False)
    study_samples = []
    for i in range(1, 496):
        x = []
        for j in range(1):
            x.append(i)
            x.append(50)
            x.append(0)
        study_samples.append(x)
    return study_samples


def Lung_Cancer():
    # 用到miRNA-disease关联3(pairs3)的数据
    # disease number:236
    all_associations = pd.read_csv ('data/all_mirna_disease_pairs.csv' , names = ['miRNA' , 'disease' , 'label'])
    all_associations = all_associations.values.tolist ()
    '''
    new_associations = []
    for i in all_associations:
        if i[1] != 50:
            new_associations.append(i)
    miRNA = []
    disease = []
    label = []
    for i in new_associations:
        miRNA.append(i[0])
        disease.append(i[1])
        label.append(i[2])
    '''
    for i in all_associations:
        if i[1] == 236 and i[2] == 1:
            i[2] = 0

    miRNA = []
    disease = []
    label = []
    for i in all_associations:
        miRNA.append (i[0])
        disease.append (i[1])
        label.append (i[2])
    data = {'miRNA': miRNA , 'disease': disease , 'label': label}
    df = pd.DataFrame (data)
    df.to_csv ('data/all_mirna_disease_pairs2.csv' , index = False)
    # data = {'miRNA': miRNA, 'disease': disease, 'label': label}
    # df = pd.DataFrame(data)
    # df.to_csv('data/all_mirna_disease_pairs1.csv', index = False)
    study_samples = []
    for i in range (1 , 496):
        x = []
        for j in range (1):
            x.append (i)
            x.append (236)
            x.append (0)
        study_samples.append (x)
    return study_samples


def Esophageal_Neoplasms():
    # 用到miRNA-disease关联4(pairs4)的数据
    # disease number:126
    all_associations = pd.read_csv ('data/all_mirna_disease_pairs.csv' , names = ['miRNA' , 'disease' , 'label'])
    all_associations = all_associations.values.tolist ()
    '''
    new_associations = []
    for i in all_associations:
        if i[1] != 50:
            new_associations.append(i)
    miRNA = []
    disease = []
    label = []
    for i in new_associations:
        miRNA.append(i[0])
        disease.append(i[1])
        label.append(i[2])
    '''
    for i in all_associations:
        if i[1] == 126 and i[2] == 1:
            i[2] = 0

    miRNA = []
    disease = []
    label = []
    for i in all_associations:
        miRNA.append (i[0])
        disease.append (i[1])
        label.append (i[2])
    data = {'miRNA': miRNA , 'disease': disease , 'label': label}
    df = pd.DataFrame (data)
    df.to_csv ('data/all_mirna_disease_pairs3.csv' , index = False)
    # data = {'miRNA': miRNA, 'disease': disease, 'label': label}
    # df = pd.DataFrame(data)
    # df.to_csv('data/all_mirna_disease_pairs1.csv', index = False)
    study_samples = []
    for i in range (1 , 496):
        x = []
        for j in range (1):
            x.append (i)
            x.append (126)
            x.append (0)
        study_samples.append (x)
    return study_samples


def Pancreatic_Neoplasms():
    # 用到miRNA-disease关联4(pairs4)的数据
    # disease number:307
    all_associations = pd.read_csv ('data/all_mirna_disease_pairs.csv' , names = ['miRNA' , 'disease' , 'label'])
    all_associations = all_associations.values.tolist ()

    '''
    new_associations = []
    for i in all_associations:
        if i[1] != 50:
            new_associations.append(i)
    miRNA = []
    disease = []
    label = []
    for i in new_associations:
        miRNA.append(i[0])
        disease.append(i[1])
        label.append(i[2])
    '''
    for i in all_associations:
        if i[1] == 307 and i[2] == 1:
            i[2] = 0

    miRNA = []
    disease = []
    label = []
    for i in all_associations:
        miRNA.append (i[0])
        disease.append (i[1])
        label.append (i[2])
    # data = {'miRNA': miRNA , 'disease': disease , 'label': label}
    # df = pd.DataFrame (data)
    # df.to_csv ('data/all_mirna_disease_pairs4.csv' , index = False)
    # data = {'miRNA': miRNA, 'disease': disease, 'label': label}
    # df = pd.DataFrame(data)
    # df.to_csv('data/all_mirna_disease_pairs1.csv', index = False)
    study_samples = []
    for i in range (1 , 496):
        x = []
        for j in range (1):
            x.append (i)
            x.append (307)
        study_samples.append (x)            x.append (0)

    return study_samples


# Pancreatic_Neoplasms()
