import numpy as np
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io
import pickle
import numpy.ma as ma
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# random.seed(42)
# np.random.seed(42)

## p_selectfeatrues表示是否采用特征筛选以及采用何种特征筛选方法
p_selectfeatrues = 'maxmin'
# p_selectfeatrues = 'fisher'
# p_selectfeatrues = False

## p_combat表示是否采用combat方法进行站点协调
p_combat = True
# p_combat = False

## p_dataset表示使用哪一个数据集
# p_dataset = 'rest-meta-mdd'
# p_dataset = 'first_episode_and_recurrent_mdd'
p_dataset = 'all-rest-meta-mdd'

## p_ROI表示使用哪一个模板
p_ROI = 'CC200'
# p_ROI = 'AAL'

## p_augmentation表示是否采用数据增强，采用数据增强的增强样本比例是多少
# p_augmentation = False
p_augmentation = 1

## p_Method表示使用哪一种方法
p_Method = 'ResDAE'

# site = 7, 9, 11, 19, 20, 21
# p_Method, p_site = 'ResDAE(every site)', 7
# p_Method = 'ResDAE(no Res)'
# p_Method = 'SVM'
# p_Method = 'RF'
# p_Method = 'FCNN'
# p_Method = '1DCNN'
# p_Method = 'self-attention'

p_threshold = 0.5
if p_ROI == 'CC200':
    roi_start, roi_end = 228, 428
else:
    # AAL
    roi_start, roi_end = 0, 116

if p_dataset == 'rest-meta-mdd':
    # 假设数据存储在名为data的numpy数组中，每个被试的数据都是一维的，共有616个被试，每个被试有19900个特征
    mdd_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/MDD"  # cc200'#path to time series data
    hc_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/HC"
    mdd_flist = sorted(os.listdir(mdd_data_main_path))
    hc_flist = sorted(os.listdir(hc_data_main_path))
    df = scipy.io.loadmat('./Data/Stat_Sub_Info_189RecurrentMDDvs427NC.mat')
    sub_id, sex, age, DX = df['SubID'], df['Sex'].squeeze(), df['Age'].squeeze(), df['Dx'].squeeze()
    sub_id = np.array([sub_id[i] for i in range(len(sub_id))]).squeeze()
    site = np.array([int(sub_id[i][0].split('-')[0].lstrip('S')) for i in range(len(sub_id))])
    fdict = {'Sub_ID': sub_id,
             'Site': site,
             'Sex': sex,
             'Age': age,
             'Dx': DX}
    df_labels = pd.DataFrame(fdict)
    df_labels.Dx = df_labels.Dx.map({1: 1, -1: 0})
    print(len(df_labels))
    if not os.path.exists('./Data/mdd_1d_data(616)_' + p_ROI + '.pkl'):
        data = []
        labels = []
        for mdd_name in tqdm(mdd_flist):
            if 'mat' in mdd_name:
                if mdd_name.split('.')[0].split('_')[1] in df_labels['Sub_ID'].values:
                    mdd_data = scipy.io.loadmat(os.path.join(mdd_data_main_path, mdd_name))['ROISignals'][:, roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(mdd_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        mdd_1d = ma.masked_where(m, corr).compressed()
                        data.append(mdd_1d)
                        labels.append(1)
        for hc_name in tqdm(hc_flist):
            if 'mat' in hc_name:
                if hc_name.split('.')[0].split('_')[1] in df_labels['Sub_ID'].values:
                    hc_data = scipy.io.loadmat(os.path.join(hc_data_main_path, hc_name))['ROISignals'][:, roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(hc_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        hc_1d = ma.masked_where(m, corr).compressed()
                        data.append(hc_1d)
                        labels.append(0)
        data = np.array(data)# 你的数据
        pickle.dump(data, open('./Data/mdd_1d_data(616)_' + p_ROI + '.pkl', 'wb'))
        # 假设你已经将每个被试的标签存储在名为labels的numpy数组中，其中1表示阳性样本，0表示阴性样本
        # labels = np.array(labels)# 你的标签
        pickle.dump(df_labels, open('./Data/mdd_1d_cov(616)_' + p_ROI + '.pkl', 'wb'))
        print(data.shape, fdict['Dx'].shape)
    data = pickle.load(open('./Data/mdd_1d_data(616)_' + p_ROI + '.pkl', 'rb'))
    cov = pickle.load(open('./Data/mdd_1d_cov(616)_' + p_ROI + '.pkl', 'rb'))
    labels, site, age, sex = cov['Dx'].values, cov['Site'].values, cov['Age'].values, cov['Sex'].values

elif p_dataset == 'all-rest-meta-mdd':
    # 假设你的数据存储在名为data的numpy数组中，每个被试的数据都是一维的，共有616个被试，每个被试有19900个特征
    mdd_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/MDD"  # cc200'#path to time series data
    hc_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/HC"
    mdd_flist = sorted(os.listdir(mdd_data_main_path))
    hc_flist = sorted(os.listdir(hc_data_main_path))
    df = scipy.io.loadmat('./Data/Cov.mat')
    sub_id = pd.read_csv("./Data/df.csv").iloc[0, 1:].values
    site, DX, age, sex = df['data'][0, :], df['data'][1, :], df['data'][2, :],df['data'][3, :]
    fdict = {'Sub_ID': sub_id,
             'Site': site,
             'Sex': sex,
             'Age': age,
             'Dx': DX}
    df_labels = pd.DataFrame(fdict)
    print(len(df_labels))
    if not os.path.exists('./Data/mdd_1d_data(1611)_' + p_ROI + '.pkl'):
        data = []
        labels = []
        for mdd_name in tqdm(mdd_flist):
            if 'mat' in mdd_name:
                id = mdd_name.split('.')[0]
                id = id.split('_')[1]
                id = id[0] + '0' + id[1:] if id[2] == '-' else id
                if id in df_labels['Sub_ID'].values:
                    mdd_data = scipy.io.loadmat(os.path.join(mdd_data_main_path, mdd_name))['ROISignals'][:,
                               roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(mdd_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        mdd_1d = ma.masked_where(m, corr).compressed()
                        data.append(mdd_1d)
                        labels.append(1)
        for hc_name in tqdm(hc_flist):
            if 'mat' in hc_name:
                id = hc_name.split('.')[0]
                id = id.split('_')[1]
                id = id[0] + '0' + id[1:] if id[2] == '-' else id
                if id in df_labels['Sub_ID'].values:
                    hc_data = scipy.io.loadmat(os.path.join(hc_data_main_path, hc_name))['ROISignals'][:,
                              roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(hc_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        hc_1d = ma.masked_where(m, corr).compressed()
                        data.append(hc_1d)
                        labels.append(0)
        data = np.array(data)  # 你的数据
        pickle.dump(data, open('./Data/mdd_1d_data(1611)_' + p_ROI + '.pkl', 'wb'))
        # 假设你已经将每个被试的标签存储在名为labels的numpy数组中，其中1表示阳性样本，0表示阴性样本
        # labels = np.array(labels)  # 你的标签
        pickle.dump(fdict, open('./Data/mdd_1d_cov(1611)_' + p_ROI + '.pkl', 'wb'))

        print(data.shape, fdict['Dx'].shape)
    data = pickle.load(open('./Data/mdd_1d_data(1611)_' + p_ROI + '.pkl', 'rb'))
    cov = pickle.load(open('./Data/mdd_1d_cov(1611)_' + p_ROI + '.pkl', 'rb'))
    labels, site, age, sex = cov['Dx'], cov['Site'], cov['Age'], cov['Sex']

elif p_dataset == 'first_episode_and_recurrent_mdd':
    # 假设数据存储在名为data的numpy数组中，每个被试的数据都是一维的，共有191个被试，每个被试有19900个特征
    mdd_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/MDD"  # cc200'#path to time series data
    hc_data_main_path = "./Data/meta-MDD/ROISignals_FunImgARglobalCWF/HC"
    mdd_flist = sorted(os.listdir(mdd_data_main_path))
    hc_flist = sorted(os.listdir(hc_data_main_path))
    df = scipy.io.loadmat('./Data/Stat_Sub_Info_119FirstEpisodeDrugNaiveMDDvs72RecurrentMDD.mat')
    sub_id, sex, age, DX = df['SubID'], df['Sex'].squeeze(), df['Age'].squeeze(), df['FirstEpisodeScore'].squeeze()
    sub_id = np.array([sub_id[i] for i in range(len(sub_id))]).squeeze()
    site = np.array([int(sub_id[i][0].split('-')[0].lstrip('S')) for i in range(len(sub_id))])
    fdict = {'Sub_ID': sub_id,
             'Site': site,
             'Sex': sex,
             'Age': age,
             'Dx': DX}
    df_labels = pd.DataFrame(fdict)
    df_labels.Dx = df_labels.Dx.map({1: 1, -1: 0})
    print(len(df_labels))
    if not os.path.exists('./Data/mdd_1d_data(191)_' + p_ROI + '.pkl'):
        data = []
        labels = []
        for mdd_name in tqdm(mdd_flist):
            if 'mat' in mdd_name:
                if mdd_name.split('.')[0].split('_')[1] in df_labels['Sub_ID'].values:
                    mdd_data = scipy.io.loadmat(os.path.join(mdd_data_main_path, mdd_name))['ROISignals'][:,
                               roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(mdd_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        mdd_1d = ma.masked_where(m, corr).compressed()
                        data.append(mdd_1d)
                        labels.append(1)
        for hc_name in tqdm(hc_flist):
            if 'mat' in hc_name:
                if hc_name.split('.')[0].split('_')[1] in df_labels['Sub_ID'].values:
                    hc_data = scipy.io.loadmat(os.path.join(hc_data_main_path, hc_name))['ROISignals'][:,
                              roi_start:roi_end]
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(hc_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        hc_1d = ma.masked_where(m, corr).compressed()
                        data.append(hc_1d)
                        labels.append(0)
        data = np.array(data)  # 你的数据
        pickle.dump(data, open('./Data/mdd_1d_data(191)_' + p_ROI + '.pkl', 'wb'))
        # 假设已经将每个被试的标签存储在名为labels的numpy数组中，其中1表示阳性样本，0表示阴性样本
        # labels = np.array(labels)# 你的标签
        pickle.dump(df_labels, open('./Data/mdd_1d_cov(191)_' + p_ROI + '.pkl', 'wb'))
        print(data.shape, fdict['Dx'].shape)
    data = pickle.load(open('./Data/mdd_1d_data(191)_' + p_ROI + '.pkl', 'rb'))
    cov = pickle.load(open('./Data/mdd_1d_cov(191)_' + p_ROI + '.pkl', 'rb'))
    labels, site, age, sex = cov['Dx'].values, cov['Site'].values, cov['Age'].values, cov['Sex'].values

else:
    # 假设你的数据存储在名为data的numpy数组中，每个被试的数据都是一维的，共有616个被试，每个被试有19900个特征
    mdd_data_main_path = "./Data/SRPBS_"+p_ROI+"/HC"  # cc200'#path to time series data
    hc_data_main_path = "./Data/SRPBS_"+p_ROI+"/MDD"
    mdd_flist = os.listdir(mdd_data_main_path)
    hc_flist = os.listdir(hc_data_main_path)
    df = pd.read_excel('./Data/SRPBS_MDD_444.xls')
    sub_id, sex, age, DX = df['participant_id'].values, df['sex'].values, df['age'].values, df['diag'].values
    fdict = {'Sub_ID': sub_id,
             'Sex': sex,
             'Age': age,
             'Dx': DX}
    df_labels = pd.DataFrame(fdict)
    df_labels.Dx = df_labels.Dx.map({2: 1, 0: 0})
    print(len(df_labels))
    if not os.path.exists('./Data/mdd_1d_data(444)_' + p_ROI + '.pkl'):
        data = []
        labels = []
        for mdd_name in tqdm(mdd_flist):
            if 'mat' in mdd_name:
                sub_id = mdd_name.split('.')[0]
                sub_id = sub_id.split('_')[1]
                sub_id = sub_id[:3] + '-' + sub_id[3:]
                if sub_id in df_labels['Sub_ID'].values:
                    mdd_data = scipy.io.loadmat(os.path.join(mdd_data_main_path, mdd_name))['ROISignals']
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(mdd_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        mdd_1d = ma.masked_where(m, corr).compressed()
                        data.append(mdd_1d)
                        labels.append(1)
        for hc_name in tqdm(hc_flist):
            if 'mat' in hc_name:
                sub_id = hc_name.split('.')[0]
                sub_id = sub_id.split('_')[1]
                sub_id = sub_id[:3] + '-' + sub_id[3:]
                if sub_id in df_labels['Sub_ID'].values:
                    hc_data = scipy.io.loadmat(os.path.join(hc_data_main_path, hc_name))['ROISignals']
                    with np.errstate(invalid="ignore"):
                        corr = np.nan_to_num(np.corrcoef(hc_data.T))
                        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
                        m = ma.masked_where(mask == 1, mask)
                        hc_1d = ma.masked_where(m, corr).compressed()
                        data.append(hc_1d)
                        labels.append(0)
        data = np.array(data)  # 你的数据
        pickle.dump(data, open('./Data/mdd_1d_data(444)_' + p_ROI + '.pkl', 'wb'))
        # 已经将每个被试的标签存储在名为labels的numpy数组中，其中1表示阳性样本，0表示阴性样本
        labels = np.array(labels)  # 你的标签
        pickle.dump(labels, open('./Data/mdd_1d_labels(444)_' + p_ROI + '.pkl', 'wb'))
        print(data.shape, labels.shape)
    data = pickle.load(open('./Data/mdd_1d_data(444)_' + p_ROI + '.pkl', 'rb'))
    labels = pickle.load(open('./Data/mdd_1d_labels(444)_' + p_ROI + '.pkl', 'rb'))


datas = np.array(data)
print(datas.shape)

if p_combat == True:
    # combat实现站点协调
    from neurocombat_sklearn import CombatModel
    label = np.reshape(labels, (labels.shape[0], 1))
    sites = np.reshape(site, (site.shape[0], 1))
    sexs = np.reshape(sex, (sex.shape[0], 1))
    ages = np.reshape(age, (age.shape[0], 1))
    combat = CombatModel()
    combat.fit(datas,
            sites = sites,
            discrete_covariates = np.concatenate((label, sexs), axis = 1),
            continuous_covariates = ages)
    datas = combat.transform(datas, sites = sites,
                             discrete_covariates = np.concatenate((label, sexs), axis = 1),
                             continuous_covariates = ages)

num_featrues = datas.shape[1]
num_subjects = datas.shape[0]

def print_log(txt):  # 保存日志，以备查用
    if p_augmentation == False and p_combat != False and p_selectfeatrues != False and p_Method == 'ResDAE':
        log_file = r"./Data/log/ResDAE/log_" + str(
            num_subjects) + "_MDD_ResDAE(no_Aug)_" + p_ROI + '_' + current_time + ".txt"
    elif p_augmentation != False and p_combat == False and p_selectfeatrues != False and p_Method == 'ResDAE':
        log_file = r"./Data/log/ResDAE/log_" + str(
            num_subjects) + "_MDD_ResDAE(no_Combat)_" + p_ROI + '_' + current_time + ".txt"
    elif p_augmentation != False and p_combat != False and p_selectfeatrues == False and p_Method == 'ResDAE':
        log_file = r"./Data/log/ResDAE/log_" + str(
            num_subjects) + "_MDD_ResDAE(no_FS)_" + p_ROI + '_' + current_time + ".txt"
    elif p_augmentation != False and p_combat != False and p_selectfeatrues == False and p_Method == 'ResDAE(no Res)':
        log_file = r"./Data/log/ResDAE/log_" + str(
            num_subjects) + "_MDD_ResDAE(no_Res)_" + p_ROI + '_' + current_time + ".txt"
    else:
        log_file = r"./Data/log/ResDAE/log_" + str(num_subjects) + "_MDD_" + p_Method + "_" + p_ROI + '_' + current_time + ".txt"
    print(txt)
    f = open(log_file, "a")
    f.write(txt)
    f.write("\n")
    f.close()

print_log("*****List of patameters****")
print_log(f"dataset: {p_dataset}")
print_log(f"ROI atlas: {p_ROI}")
print_log(f"Method's name: {p_Method}")
print_log(f"Augmentation: {p_augmentation}")
print_log(f"threshold: {p_threshold}")
print_log(f"featrue select: {p_selectfeatrues}")
print_log(f"combat: {p_combat}")

if p_selectfeatrues == 'maxmin':
    def get_featrues(datas, labels):
        avg = []
        for ie in range(datas.shape[1]):
            avg.append(np.mean(datas[:, ie]))
        avg = np.array(avg)
        highs = avg.argsort()[-num_featrues // 4:][::-1]
        lows = avg.argsort()[:num_featrues // 4][::-1]
        regions = np.concatenate((highs, lows), axis=0)
        # data = np.array([datas[i][regions] for i in range(datas.shape[0])])
        return regions
elif p_selectfeatrues == 'fisher':
    def get_featrues(datas, labels):
        from sklearn.feature_selection import SelectKBest, f_classif
        # datas是特征矩阵，labels是标签
        selector = SelectKBest(score_func=f_classif, k=num_featrues // 2) # 在此处选择 fisher score
        X_new = selector.fit_transform(datas, labels)
        regions = selector.get_support(indices=True)
        # data = np.array([datas[i][regions] for i in range(datas.shape[0])])
        return regions

print(data.shape)


import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        y = torch.from_numpy(np.array(self.labels[index])).float()
        return x, y
    
    def __len__(self):
        return len(self.data)
    
import torch
import torch.nn as nn


if p_Method == 'ResDAE' or p_Method == 'ResDAE(every site)':
    class AutoencoderClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(AutoencoderClassifier, self).__init__()
            self.fc_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                                            nn.Linear(hidden_dim, hidden_dim)#, nn.Tanh(), nn.BatchNorm1d(hidden_dim)
                                            )
            self.enc_res_layer = nn.Linear(input_dim, hidden_dim)
            self.fc_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, input_dim)#, nn.Tanh(), nn.BatchNorm1d(input_dim)
            )
            self.dec_res_layer = nn.Linear(hidden_dim, input_dim)
            self.classifier = nn.Sequential(
                # nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            fc_enc = self.fc_encoder(x)
            res_enc = self.enc_res_layer(x)
            x_enc = fc_enc + res_enc
            x_enc = torch.tanh(x_enc)
            y = self.classifier(x_enc)
            fc_dec = self.fc_decoder(x_enc)
            res_dec = self.dec_res_layer(x_enc)
            x_hat = fc_dec + res_dec
            return x_hat, y.squeeze()

elif p_Method == 'ResDAE(no Res)':
    class AutoencoderClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(AutoencoderClassifier, self).__init__()
            self.fc_encoder = nn.Linear(input_dim, hidden_dim)
            self.fc_decoder = nn.Linear(hidden_dim, input_dim)
            self.classifier = nn.Sequential(
                # nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            # Encode
            z = self.fc_encoder(x)
            z = torch.tanh(z)
            # Decode
            x_hat = self.fc_decoder(z)
            # Classify
            y = self.classifier(z)
            return x_hat, y.squeeze()

elif p_Method == 'self-attention':
    class SelfAttention(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(SelfAttention, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # Embedding
            x = self.embedding(x).unsqueeze(1)

            # Permute and reshape input for self-attention layer
            x = x.permute(1, 0, 2)
            x, _ = self.attention(x, x, x)
            x = x.permute(1, 0, 2)
            x = torch.mean(x, dim=1)  # Average across sequence dimension

            # Classification layer
            x = self.fc(x)
            return x.squeeze()

elif p_Method == '1DCNN':
    class CNN_1D(nn.Module):
        def __init__(self, input_size, num_classes):
            super(CNN_1D, self).__init__()
            # self.hidden_size = 4973#832 if p_selectfeatrues != False else 4973
            # self.hidden_size = 1666
            # self.hidden_size = 2486
            self.hidden_size = 832
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(64 * self.hidden_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x.squeeze()

elif p_Method == 'FCNN':
    class FCNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.dropout(self.fc1(x)))
            x = torch.relu(self.dropout(self.fc2(x)))
            x = self.dropout(self.fc3(x))
            return x.squeeze()

device = 'cuda'


# 定义训练函数
def train(model, optimizer, criterion, dataloader, device = device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        x_hat, outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))+nn.MSELoss()(x_hat, inputs.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# 定义验证函数
def validate(model, criterion, dataloader, device = device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    y_true, y_score, all_preds = [], [], []
    for inputs, targets in dataloader:
        with torch.no_grad():
            x_hat, outputs = model(inputs.to(device))
            # print("outputs shape: ", outputs.shape)

            clf_loss = criterion(outputs, targets.to(device))
            ae_loss = nn.MSELoss()(x_hat, inputs.to(device))
            loss = clf_loss+ae_loss
            running_loss += loss.item() * inputs.size(0)

            proba = torch.sigmoid(outputs).detach().cpu().numpy()
            y_true += targets.tolist()
            y_score += proba.tolist()
            predss = np.ones_like(proba, dtype=np.int32)
            # predss[proba < 0.6826] = 0
            predss[proba < p_threshold] = 0

            all_preds += predss.tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, all_preds).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)

    return running_loss / len(dataloader.dataset), accuracy, sensitivity, specificty, y_true, y_score, all_preds


def other_train(model, optimizer, criterion, dataloader, device = device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def other_valid(model, criterion, dataloader, device = device):
    model.eval()
    running_loss = 0.0
    y_true, y_score, all_preds = [], [], []
    for inputs, targets in dataloader:
        with torch.no_grad():
            outputs = model(inputs.to(device))
            # print("outputs shape: ", outputs.shape)

            loss = criterion(outputs, targets.to(device))
            running_loss += loss.item() * inputs.size(0)

            proba = torch.sigmoid(outputs).detach().cpu().numpy()
            y_true += targets.tolist()
            y_score += proba.tolist()
            predss = np.ones_like(proba, dtype=np.int32)
            # predss[proba < 0.6826] = 0
            predss[proba < p_threshold] = 0

            all_preds += predss.tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, all_preds).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)

    return running_loss / len(dataloader.dataset), accuracy, sensitivity, specificty, y_true, y_score, all_preds


def save_model(model,repeat, fold, p_dataset, p_Method, p_ROI):
    path = os.path.join("./no_FS/best_model_on_everyfold", p_dataset + '_' + p_Method + '_' + p_ROI)
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = f"{path}/model_repeat{repeat}_fold{fold}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Saved model fold {fold} to {filename}")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score
import torch.optim as optim
import copy
import sys


from collections import Counter

if p_Method in ['ResDAE', 'ResDAE(no Res)']:
    num_epochs = 30
    all_rp_res = []
    all_fpr = []
    all_tpr = []
    thresholds = []

    for rp in range(5):
        # 加载数据集并进行五次五折交叉验证
        fprs, tprs, aucs = [], [], []
        crossval_acc, crossval_sen, crossval_spe = [], [], []
        all_metric = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kf.split(datas, labels)):
            print(f"Fold {fold+1}")
            train_data, train_labels, val_data, val_labels = datas[train_idx], labels[train_idx], datas[val_idx], labels[val_idx]
            if p_selectfeatrues != False:
                regs = get_featrues(train_data, train_labels)
            else:
                regs = np.array(range(0, num_featrues))
            train_data, val_data = train_data[:, regs], val_data[:, regs]
            if p_augmentation:
                # 对数据集进行过采样
                from imblearn.over_sampling import SMOTE

                # smote = SMOTE(sampling_strategy='minority',k_neighbors=5)
                smote = SMOTE(k_neighbors=5)
                # smote进行数据增强之后，增加的样本数据全部都放在原数据的末尾
                aug_data, aug_labels = smote.fit_resample(train_data, train_labels)
                # 使用参数p_augmentation控制增加样本的数量
                aug_num = (int) ((aug_data.shape[0] - train_data.shape[0]) * p_augmentation)
                train_data = aug_data[:train_data.shape[0] + aug_num]
                train_labels = aug_labels[:train_data.shape[0] + aug_num]
                # # 对数据集进行欠采样
                # from imblearn.under_sampling import RandomUnderSampler
                # undersample = RandomUnderSampler(random_state=42)
                # train_data, train_labels = undersample.fit_resample(train_data, train_labels)
            print(Counter(train_labels))
            train_dataset = MyDataset(train_data, train_labels)
            val_dataset = MyDataset(val_data, val_labels)
            # 将数据集划分为训练集和验证集
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
            # 创建模型和优化器
            # model = DRSN(in_channels=1, num_classes=2).to(device)
            if p_selectfeatrues:
                model = AutoencoderClassifier(num_featrues // 2, num_featrues // 4, 1).to(device)
            else:
                model = AutoencoderClassifier(num_featrues, num_featrues // 2, 1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            # 训练和验证
            best_val_acc = 0.0
            max_f1 = 0
            max_mu, min_std = 0, sys.float_info.max
            best_model = None
            for epoch in range(num_epochs):
                train_loss = train(model, optimizer, criterion, train_loader)
                val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = validate(model, criterion, val_loader)
                F1 = f1_score(y_true, all_preds)

                print(
                    f"Repeat {rp + 1} Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val Sen {val_sen:.4f}, Val Spe {val_spe:.4f}")
                mu, std = np.mean([val_acc, val_sen, val_spe]), np.std([val_acc, val_sen, val_spe])
                if F1 > max_f1:
                    max_f1 = F1
                # if std < min_std and mu > max_mu and val_acc > 0.6:
                #     max_mu = mu
                #     min_std = std
                #     best_val_acc = val_acc
                #     best_model = copy.deepcopy(model)

                if p_Method in ['ResDAE', 'ResDAE(no Res)']:
                    if std < 0.13:
                        if mu > max_mu:
                            max_mu = mu
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_model = copy.deepcopy(model)
            model = best_model if best_model is not None else model

            # if rp == 0:
            save_model(model, rp, fold, p_dataset, p_Method, p_ROI)

            val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = validate(model, criterion, val_loader)

            all_metric.append([val_acc, val_sen, val_spe])
            fpr, tpr, threshold = roc_curve(y_true, y_score)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = threshold[optimal_idx]
            thresholds.append(optimal_threshold)
            auc_score = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc_score)
            print_log(f"repeat {rp + 1} fold {fold + 1} : {val_acc, val_sen, val_spe}, auc: {auc_score:.4f}")
            print_log(f'best accuracy: {best_val_acc:.4f}')

        print_log(f"Avergae result of 5 fold: mean:{np.mean(np.array(all_metric), axis=0)} "
                  f"std:{np.std(np.array(all_metric), axis=0)}")
        print_log(f'auc: {sum(aucs) / len(aucs):.4f}')
        all_rp_res.append(np.mean(np.array(all_metric), axis=0))
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)

        everyrepeat_dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": sum(aucs) / len(aucs)}
        everyrepeat_df = pd.DataFrame(everyrepeat_dict)

        if p_augmentation == False and p_combat != False and p_selectfeatrues != False and p_Method == 'ResDAE':
            everyrepeat_df.to_csv("./Data/all_csv/ResDAE/repeat_" + str(rp + 1) + "/ResDAE(no_Aug)_ROC("
                                  + str(num_subjects) + p_ROI + ").csv", index=False)
        elif p_augmentation != False and p_combat == False and p_selectfeatrues != False and p_Method == 'ResDAE':
            everyrepeat_df.to_csv("./Data/all_csv/ResDAE/repeat_" + str(rp + 1) + "/ResDAE(no_Combat)_ROC("
                                  + str(num_subjects) + p_ROI + ").csv", index=False)
        elif p_augmentation != False and p_combat != False and p_selectfeatrues == False and p_Method == 'ResDAE':
            everyrepeat_df.to_csv("./Data/all_csv/ResDAE/repeat_" + str(rp + 1) + "/ResDAE(no_FS)_ROC("
                                  + str(num_subjects) + p_ROI + ").csv", index=False)
        elif p_augmentation != False and p_combat != False and p_selectfeatrues != False and p_Method == 'ResDAE(no Res)':
            everyrepeat_df.to_csv("./Data/all_csv/ResDAE/repeat_" + str(rp + 1) + "/ResDAE(no_Res)_ROC("
                                  + str(num_subjects) + p_ROI + ").csv", index=False)
        all_fpr.append(mean_fpr)
        all_tpr.append(mean_tpr)

    print_log(f"=" * 30)
    print_log(f"Avergae result of 5 repeat: {np.mean(np.array(all_rp_res), axis=0)} "
              f"std:{np.std(np.array(all_rp_res), axis=0)}")
    mean_fpr = np.mean(all_fpr, axis=0)
    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    best_threshold = np.mean(thresholds)
    print_log(f"best threshold: {best_threshold:.4f}")
    print_log(f"auc: {mean_auc}")
    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    if p_augmentation == False and p_combat != False and p_selectfeatrues != False and p_Method == 'ResDAE':
        df.to_csv("./Data/all_csv/Ablation_Study/ResDAE(no_Aug)_ROC("
                  + str(num_subjects) + p_ROI + ").csv", index=False)
    elif p_augmentation != False and p_combat == False and p_selectfeatrues != False and p_Method == 'ResDAE':
        df.to_csv("./Data/all_csv/Ablation_Study/ResDAE(no_Combat)_ROC("
                + str(num_subjects) + p_ROI + ").csv", index=False)
    elif p_augmentation != False and p_combat != False and p_selectfeatrues == False and p_Method == 'ResDAE':
        df.to_csv("./Data/all_csv/Ablation_Study/ResDAE(no_FS)_ROC("
                + str(num_subjects) + p_ROI + ").csv", index=False)
    elif p_augmentation != False and p_combat != False and p_selectfeatrues != False and p_Method == 'ResDAE(no Res)':
        df.to_csv("./Data/all_csv/Ablation_Study/ResDAE(no_Res)_ROC("
                + str(num_subjects) + p_ROI + ").csv", index=False)
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()

# elif p_Method == 'ResDAE(every site)':
#     for i in range(10):
#         print_log(f'repeat {i}')
#         num_epochs = 30
#         print_log(f'train on site 20 and validate on site {p_site}')
#         # 将数据量最多的站点作为训练数据
#         val_data, val_labels = datas[site == p_site], labels[site == p_site]
#         train_data, train_labels = datas[site == 20], labels[site == 20]
#         print_log(f'site 20 samples: {train_data.shape[0]}, site {p_site} samples: {val_data.shape[0]}')
#         if p_selectfeatrues != False:
#             regs = get_featrues(train_data, train_labels)
#         else:
#             regs = np.array(range(0, num_featrues))
#         train_data, val_data = train_data[:, regs], val_data[:, regs]
#         if p_augmentation:
#             # 对数据集进行过采样
#             from imblearn.over_sampling import SMOTE
#
#             smote = SMOTE(k_neighbors=5)
#             # smote进行数据增强之后，增加的样本数据全部都放在原数据的末尾
#             aug_data, aug_labels = smote.fit_resample(train_data, train_labels)
#             # 使用参数p_augmentation控制增加样本的数量
#             aug_num = (int)((aug_data.shape[0] - train_data.shape[0]) * p_augmentation)
#             train_data = aug_data[:train_data.shape[0] + aug_num]
#             train_labels = aug_labels[:train_data.shape[0] + aug_num]
#
#         print(Counter(train_labels))
#         train_dataset = MyDataset(train_data, train_labels)
#         val_dataset = MyDataset(val_data, val_labels)
#         # 将数据集划分为训练集和验证集
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
#         # 创建模型和优化器
#         if p_selectfeatrues:
#             model = AutoencoderClassifier(num_featrues // 2, num_featrues // 4, 1).to(device)
#         else:
#             model = AutoencoderClassifier(num_featrues, num_featrues // 2, 1).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
#         criterion = nn.BCEWithLogitsLoss()
#         # 训练和验证
#         best_val_acc = 0.0
#         max_f1 = 0
#         max_mu, min_std = 0, sys.float_info.max
#         best_model = None
#         for epoch in range(num_epochs):
#             train_loss = train(model, optimizer, criterion, train_loader)
#             val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = validate(model, criterion, val_loader)
#             F1 = f1_score(y_true, all_preds)
#
#             print(
#                 f"Epoch [{epoch + 1}/{num_epochs}], Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val Sen {val_sen:.4f}, Val Spe {val_spe:.4f}")
#             mu, std = np.mean([val_acc, val_sen, val_spe]), np.std([val_acc, val_sen, val_spe])
#             if F1 > max_f1:
#                 max_f1 = F1
#
#             if std < 0.13:
#                 if mu > max_mu:
#                     max_mu = mu
#                     if val_acc > best_val_acc:
#                         best_val_acc = val_acc
#                         best_model = copy.deepcopy(model)
#         model = best_model if best_model is not None else model
#         # save_model(model, 1, p_dataset, p_Method, p_ROI)
#
#         val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = validate(model, criterion, val_loader)
#
#         fpr, tpr, threshold = roc_curve(y_true, y_score)
#         auc_score = auc(fpr, tpr)
#         print_log(f"site {p_site} metric: {val_acc, val_sen, val_spe}, auc: {auc_score:.4f}")
#         print_log(f'best accuracy: {best_val_acc:.4f}')


elif p_Method in ['1DCNN', 'self-attention', 'FCNN']:
    num_epochs = 30
    all_rp_res = []
    all_fpr = []
    all_tpr = []
    thresholds = []

    for rp in range(5):
        # 加载数据集并进行五次五折交叉验证
        fprs, tprs, aucs = [], [], []
        crossval_acc, crossval_sen, crossval_spe = [], [], []
        all_metric = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kf.split(datas, labels)):
            print(f"Fold {fold + 1}")
            train_data, train_labels, val_data, val_labels = datas[train_idx], labels[train_idx], datas[val_idx], labels[
                val_idx]
            if p_selectfeatrues != False:
                regs = get_featrues(train_data, train_labels)
            else:
                regs = np.array(range(0, num_featrues))
            train_data, val_data = train_data[:, regs], val_data[:, regs]
            if p_augmentation:
                # 对数据集进行过采样
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(k_neighbors=5)
                train_data, train_labels = smote.fit_resample(train_data, train_labels)
                print(Counter(train_labels))
                # # 对数据集进行欠采样
                # from imblearn.under_sampling import RandomUnderSampler
                # undersample = RandomUnderSampler(random_state=42)
                # train_data, train_labels = undersample.fit_resample(train_data, train_labels)
            print(Counter(train_labels))
            train_dataset = MyDataset(train_data, train_labels)
            val_dataset = MyDataset(val_data, val_labels)
            # 将数据集划分为训练集和验证集
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
            # 创建模型和优化器
            if p_selectfeatrues == False:
                if p_Method == 'self-attention':
                    model = SelfAttention(num_featrues, num_featrues // 2, 1).to(device)
                elif p_Method == '1DCNN':
                    model = CNN_1D(num_featrues, 1).to(device)
                else:
                    model = FCNN(num_featrues, num_featrues // 2, 1).to(device)
            else:
                if p_Method == 'self-attention':
                    model = SelfAttention(num_featrues // 2, num_featrues // 4, 1).to(device)
                elif p_Method == '1DCNN':
                    model = CNN_1D(num_featrues // 2, 1).to(device)
                else:
                    model = FCNN(num_featrues // 2, num_featrues // 4, 1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            # 训练和验证
            best_val_acc = 0.0
            max_f1 = 0
            max_mu, min_std = 0, sys.float_info.max
            best_model = None
            for epoch in range(num_epochs):
                train_loss = other_train(model, optimizer, criterion, train_loader)
                val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = other_valid(model, criterion, val_loader)
                F1 = f1_score(y_true, all_preds)

                print(
                    f"Repeat {rp + 1} Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val Sen {val_sen:.4f}, Val Spe {val_spe:.4f}")
                mu, std = np.mean([val_acc, val_sen, val_spe]), np.std([val_acc, val_sen, val_spe])
                if F1 > max_f1:
                    max_f1 = F1
                # if std < min_std and mu > max_mu and val_acc > 0.6:
                #     max_mu = mu
                #     min_std = std
                #     best_val_acc = val_acc
                #     best_model = copy.deepcopy(model)
                if std < 0.12:
                    if mu > max_mu:
                        max_mu = mu
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model = copy.deepcopy(model)
            model = best_model if best_model is not None else model

            if rp == 0:
                save_model(model, fold, p_dataset, p_Method, p_ROI)

            val_loss, val_acc, val_sen, val_spe, y_true, y_score, all_preds = other_valid(model, criterion, val_loader)

            all_metric.append([val_acc, val_sen, val_spe])
            fpr, tpr, threshold = roc_curve(y_true, y_score)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = threshold[optimal_idx]
            thresholds.append(optimal_threshold)
            auc_score = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc_score)
            print_log(f"repeat {rp + 1} fold {fold + 1} : {val_acc, val_sen, val_spe}, auc: {auc_score:.4f}")
            print_log(f'best accuracy {best_val_acc:.4f}')

        print_log(f"Avergae result of 5 fold: mean:{np.mean(np.array(all_metric), axis=0)} "
                  f"std:{np.std(np.array(all_metric), axis=0)}")
        print_log(f'auc: {sum(aucs) / len(aucs):.4f}')
        all_rp_res.append(np.mean(np.array(all_metric), axis=0))
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
        everyrepeat_dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": sum(aucs) / len(aucs)}
        everyrepeat_df = pd.DataFrame(everyrepeat_dict)

        everyrepeat_df.to_csv("./Data/all_csv/other_method/repeat_" + str(rp + 1) + "/" + p_Method + "(aug)_ROC("
                              + str(num_subjects) + p_ROI + ").csv", index=False)
        all_fpr.append(mean_fpr)
        all_tpr.append(mean_tpr)

    print_log(f"=" * 30)
    print_log(f"Avergae result of 5 repeat: {np.mean(np.array(all_rp_res), axis=0)} "
              f"std:{np.std(np.array(all_rp_res), axis=0)}")
    mean_fpr = np.mean(all_fpr, axis=0)
    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    best_threshold = np.mean(thresholds)
    print_log(f"best threshold: {best_threshold:.4f}")
    print_log(f"auc: {mean_auc}")
    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    if p_augmentation:
        df.to_csv("./Data/all_csv/" + p_Method + "(aug)_ROC("
                  + str(num_subjects) + p_ROI + ").csv", index=False)
    else:
        df.to_csv("./Data/all_csv/" + p_Method + "(noaug)_ROC("
                  + str(num_subjects) + p_ROI + ").csv", index=False)
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()

elif p_Method in ['SVM', 'RF']:

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier


    clf = SVC(kernel = 'linear', probability=True) if p_Method == 'SVM' else RandomForestClassifier(n_estimators=100)
    all_rp_res = []
    all_fpr = []
    all_tpr = []

    for rp in range(5):
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        fprs, tprs, aucs = [], [], []
        res = []
        for kk, (train_index, test_index) in enumerate(kf.split(data, labels)):
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            if p_selectfeatrues != False:
                regs = get_featrues(train_data, train_labels)
            else:
                regs = np.array(range(0, num_featrues))
            train_data, test_data = train_data[:, regs], test_data[:, regs]
            if p_augmentation:
                # 对数据集进行欠采样
                from imblearn.under_sampling import RandomUnderSampler
                undersample = RandomUnderSampler()
                train_data, train_labels = undersample.fit_resample(train_data, train_labels)
                # #对数据集进行过采样
                # from imblearn.over_sampling import BorderlineSMOTE
                # smote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)
                # train_data, train_labels = smote.fit_resample(train_data, train_labels)

            clf.fit(train_data, train_labels)

            y_scores = clf.predict_proba(test_data)[:, 1]
            fpr, tpr, thresholds = roc_curve(test_labels, y_scores)
            auc_score = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc_score)
            pr = clf.predict(test_data)

            tn, fp, fn, tp = confusion_matrix(test_labels, pr).ravel()
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            sensitivity = (tp) / (tp + fn)
            specificty = (tn) / (tn + fp)
            res.append([accuracy, sensitivity, specificty])
            print_log(f"repeat {rp + 1} fold {kk + 1}: {[accuracy, sensitivity, specificty]}\n")

        print_log(f"average of 5 fold: mean:{np.mean(res, axis=0).tolist()} "
                  f"std:{np.std(res,axis=0).tolist()}")
        print_log(f'auc: {sum(aucs) / len(aucs):.4f}')
        all_rp_res.append(np.mean(np.array(res), axis=0))
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)

        everyrepeat_dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": sum(aucs) / len(aucs)}
        everyrepeat_df = pd.DataFrame(everyrepeat_dict)

        everyrepeat_df.to_csv("./Data/all_csv/RF_SVM/repeat_" + str(rp + 1) + "/" + p_Method + "(aug)_ROC("
                              + str(num_subjects) + p_ROI + ").csv", index=False)
        all_fpr.append(mean_fpr)
        all_tpr.append(mean_tpr)

    print_log(f"=" * 30)
    print_log(f"Avergae result of 5 repeat: {np.mean(np.array(all_rp_res), axis=0)} "
              f"std:{np.std(np.array(all_rp_res), axis=0)}")
    mean_fpr = np.mean(all_fpr, axis=0)
    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print_log(f"auc: {mean_auc}")
    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("./Data/all_csv/" + p_Method + "_ROC(" + str(num_subjects) + p_ROI + ").csv", index=False)