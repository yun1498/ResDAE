import pandas as pd
import os
import shap
import torch
import pickle
import numpy as np
import torch.nn as nn

n = 200


row_indices, col_indices = np.triu_indices(n, k=1)
feature_names = np.array([f"[{row_indices[i] + 1},{col_indices[i] + 1}]" for i in range(len(row_indices))])
print(len(feature_names))


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




class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoencoderClassifier, self).__init__()
        self.fc_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                                        nn.Linear(hidden_dim, hidden_dim)  # , nn.Tanh(), nn.BatchNorm1d(hidden_dim)
                                        )
        self.enc_res_layer = nn.Linear(input_dim, hidden_dim)
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)  # , nn.Tanh(), nn.BatchNorm1d(input_dim)
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


p_ROI = 'CC200'

# 获取数据
# data = pickle.load(open('./Data/mdd_1d_data(616)_' + p_ROI + '.pkl', 'rb'))
data = pickle.load(open('./Data/mdd_1d_data(1611)_' + p_ROI + '.pkl', 'rb'))
datas = np.array(data)
# cov = pickle.load(open('./Data/mdd_1d_cov(616)_' + p_ROI + '.pkl', 'rb'))
cov = pickle.load(open('./Data/mdd_1d_cov(1611)_' + p_ROI + '.pkl', 'rb'))
labels, site, age, sex = cov['Dx'], cov['Site'], cov['Age'], cov['Sex']



import numpy as np
from scipy.stats import ttest_ind

# # 假设样本数据为X，其中X是一个形状为（616，19900）的数组
# # 第一列到第19900列是特征
# feature_index = np.array([[70,199],
#                  [159,195],
#                  [91,191],
#                  [32,188],
#                  [50,106],
#                  [47,124],
#                  [137,148],
#                  [136,160]])
#
# for l in feature_index:
#     i, j = l[0], l[1]
#     print(str(i) + "to" + str(j))
#     index = int((400 - i) * (i - 1) / 2 + j - i - 1)
#     print(index)
#     # 将数据集按照标签类别分为两个样本
#     sample0 = datas[labels == 0, index]
#     sample1 = datas[labels == 1, index]
#
#     # 计算每个样本的样本均值、样本标准差和样本大小
#     mean0, mean1 = np.mean(sample0), np.mean(sample1)
#     std0, std1 = np.std(sample0), np.std(sample1)
#     n0, n1 = len(sample0), len(sample1)
#
#     # 计算双样本t检验的t值和P值
#     t, p = ttest_ind(sample0, sample1)
#
#     # 打印结果
#     print(f'样本0的平均值：{mean0:.2f}')
#     print(f'样本1的平均值：{mean1:.2f}')
#     print(f'样本0的标准差：{std0:.2f}')
#     print(f'样本1的标准差：{std1:.2f}')
#     print(f'样本0的大小：{n0}')
#     print(f'样本1的大小：{n1}')
#     print(f't值：{t:.2f}')
#     print(f'P值：{p:.4f}')
#     print('-----------------------------')




from neurocombat_sklearn import CombatModel

label = np.reshape(labels, (labels.shape[0], 1))
sites = np.reshape(site, (site.shape[0], 1))
sexs = np.reshape(sex, (sex.shape[0], 1))
ages = np.reshape(age, (age.shape[0], 1))
combat = CombatModel()
combat.fit(datas,
           sites=sites,
           discrete_covariates=np.concatenate((label, sexs), axis=1),
           continuous_covariates=ages)
datas = combat.transform(datas, sites=sites,
                         discrete_covariates=np.concatenate((label, sexs), axis=1),
                         continuous_covariates=ages)

num_featrues = datas.shape[1]
repeats, folds = 5, 5
if not os.path.exists('./max_list_1611.npz'):
    max_list = []
    for rp in range(repeats):
        for fold in range(folds):
            print(f'process repeat{rp + 1} fold{fold + 1} ...')
            model = AutoencoderClassifier(num_featrues // 2, num_featrues // 4, 1)
            # 加载训练好的PyTorch模型
            # model.load_state_dict(torch.load(f'./best_model_on_everyfold/rest-meta-mdd_ResDAE_CC200/model_fold{fold}.pt'))
            model.load_state_dict(torch.load(f'./no_FS/best_model_on_everyfold/all-rest-meta-mdd_ResDAE_CC200/model_repeat{rp}_fold{fold}.pt'))

            class newmodel(nn.Module):
                def __init__(self, model):
                    super(newmodel, self).__init__()
                    self.model = model

                def forward(self, x):
                    output = self.model(x)[1]
                    return output.unsqueeze(1)

            model = newmodel(model)
            model.to('cuda:0')
            model.eval()


            # from sklearn.feature_selection import SelectKBest, f_classif
            # # datas是特征矩阵，labels是标签
            # selector = SelectKBest(score_func=f_classif, k=num_featrues // 2)  # 在此处选择 fisher score
            # X_new = selector.fit_transform(datas, labels)
            # regs = selector.get_support(indices=True)
            # datas = np.array([datas[i][regs] for i in range(datas.shape[0])])



            from sklearn.model_selection import StratifiedKFold
            import matplotlib
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from torch.autograd import Variable


            train_data, val_data, train_labels, val_labels = train_test_split(datas, labels, test_size=0.4, random_state=42)
            regs = get_featrues(train_data, train_labels)
            datas = datas[:, regs]
            train_data, val_data = train_data[:, regs], val_data[:, regs]

            # train_data = torch.tensor(train_data).float().to('cuda:0')
            # val_data = torch.tensor(val_data).float().to('cuda:0')
            # train_labels = torch.tensor(train_labels).view(-1, 1).float()
            # val_labels = torch.tensor(val_labels).view(-1, 1).float()

            # 初始化SHAP解释器
            torch.set_grad_enabled(True)
            print("初始化SHAP解释器中......")
            # explainer = shap.DeepExplainer(model, Variable( torch.from_numpy( val_data.astype('float32') ) ).to('cuda:0'))
            # explainer = shap.DeepExplainer(model, Variable( torch.from_numpy( datas.astype('float32') ) ).to('cuda:0'))
            explainer = shap.DeepExplainer(model, Variable( torch.from_numpy( train_data.astype('float32') ) ).to('cuda:0'))


            # 计算SHAP值
            print("计算SHAP值中......")
            #background_samples = shap.sample(val_data, 10)
            # shap_values:(124, 9950)
            # shap_values = explainer.shap_values(Variable( torch.from_numpy( val_data.astype('float32') ) ).to('cuda:0'))
            # shap_values = explainer.shap_values(Variable( torch.from_numpy( datas.astype('float32') ) ).to('cuda:0'))
            shap_values = explainer.shap_values(Variable( torch.from_numpy( train_data.astype('float32') ) ).to('cuda:0'))

            # 可视化SHAP值
            print("可视化SHAP值中......")
            plt.figure(figsize=(10, 10), dpi=500)
            # plt.title(f'fold {fold} dot')
            shap.summary_plot(shap_values, train_data.astype('float32'), feature_names[regs], plot_type= "dot",show=False,
                              max_display=20, plot_size=(10, 10))
            # shap.summary_plot(shap_values, val_data.astype('float32'), feature_names[regs], plot_type= "dot",show=False,
            #                   max_display=20, plot_size=(10, 10))
            # shap.summary_plot(shap_values, datas.astype('float32'), feature_names[regs], plot_type= "dot",show=False,
            #                   max_display=20, plot_size=(10, 10))
            plt.tight_layout()
            plt.savefig(f"./shap_values/1611samples/1611_shape_values_dot_repeat{rp + 1}_fold{fold + 1}.tif", dpi=500, format='tif')
            plt.show()

            plt.figure(figsize=(10, 10), dpi=500)
            # plt.title(f'fold {fold} bar')
            shap.summary_plot(shap_values, train_data.astype('float32'), feature_names[regs], plot_type="bar", show=False,
                              max_display=20, plot_size=(10, 10))
            # shap.summary_plot(shap_values, val_data.astype('float32'), feature_names[regs], plot_type="bar", show=False,
            #                   max_display=20, plot_size=(10, 10))
            # shap.summary_plot(shap_values, datas.astype('float32'), feature_names[regs], plot_type="bar", show=False,
            #                   max_display=20, plot_size=(10, 10))
            plt.tight_layout()
            plt.savefig(f"./shap_values/1611samples/1611_shape_values_bar_repeat{rp}_fold{fold + 1}.tif", dpi=500, format='tif')
            plt.show()

            # shap_values = explainer.shap_values(Variable( torch.from_numpy( val_data.astype('float32') ) ).to('cuda:0'))
            np.savez(f'./shap_values/1611samples/1611_shape_values_repeat{rp + 1}_fold{fold + 1}.npz', shap_values)
            shap_abs_mean = np.array([np.mean(np.abs(shap_values[:, i])) for i in range(shap_values.shape[1])])

            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(50, len(feature_order)):]
            # max_list.append(feature_names[regs][feature_order])
            max_list.append(feature_names[regs][feature_order])
    arrays = np.array(max_list)
    np.savez('./max_list_1611.npz', arrays)
    # np.savez('./shap_values/1611samples/max_list_1611.npz', arrays)
else:
    # arrays = np.load('./shap_values/1611samples/max_list_1611.npz')['arr_0']
    arrays = np.load('./max_list_1611.npz')['arr_0']

k = 10  # 至少出现在 k 个数组中

# 使用字典统计元素的出现次数
counts = {}
for arr in arrays:
    for x in set(arr):
        counts[x] = counts.get(x, 0) + 1

# 打印出现次数至少为 k 的元素
result = [x for x, count in counts.items() if count >= k]
if result:
    print(f"在至少 {k} 个数组中出现过的元素有：")
    print(result)
else:
    print(f"没有元素在至少 {k} 个数组中出现过。")

# plt.figure(figsize=(10, 10), dpi=500)
# plt.title(f'fold {fold} dot')
# shap.summary_plot(shap_values, datas.astype('float32'), feature_names[regs], plot_type= "dot",show=False,
#                   max_display= 50, plot_size=(10, 10))
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 10), dpi=500)
# plt.title(f'fold {fold} bar')
# shap.summary_plot(shap_values, datas.astype('float32'), feature_names[regs], plot_type= "bar",show=False,
#                   max_display= 50, plot_size=(10, 10))
# plt.tight_layout()
# plt.show()


# #kernelexplainer
#
# f = lambda x: model(Variable(torch.from_numpy(x.astype('float32')))).detach().numpy()
# explainer = shap.KernelExplainer(f, val_data.astype(float))
# shap_values = explainer.shap_values(val_data.astype(float))
# shap.initjs()
#
# shap.summary_plot(shap_values[0], val_data.astype(float), feature_names[regs])

