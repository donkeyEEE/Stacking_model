import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import numpy as np
import matplotlib.font_manager as font_manager
import deepchem as dc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score, \
    roc_auc_score
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.decomposition import PCA, NMF

class Args:
    def __init__(self, train, test, predict_df=None,
                 AFP=False, RF=False, MPNN=False, SVR=False, GAT=False,
                 save_r=False, DNN=False,
                 MLR=False, RF2=False, SVR2=False,
                 plot=False, AD_FP=False,
                 S_C=[0.8, 0.4]):

        # 储存训练集和测试集
        # 以及位置化合物数据集
        self.train = train
        self.test = test
        self.predict_df = predict_df

        # 第一层模型列表
        self.model_lis = [AFP, RF, MPNN, SVR]
        self.GAT = GAT
        self.DNN = DNN
        # 第二层模型列表
        self.model_lis_L2 = [MLR, RF2, SVR2]

        # 其他设置
        # 是否保存预测结果数据集
        # 是否采用适用域处理
        # S_C列表传递阈值S和C
        # 是否作测试集预测值和真实值的对比图
        self.save_r = save_r
        self.AD_FP = AD_FP
        self.S_C = S_C
        self.plot = plot
        self.ECFP_Params = [4096,2]


def get_maccs_fingerprint(smiles):
    if True:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp
    # 计算化合物间的谷本系数


def get_tanimoto_similarity(fp1, fp2):
    if True:
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)


class ADFP_AC:
    def __init__(self, train, test, S, C):
        self.train = train
        self.test = test
        self.threshold = S
        self.C = C

    # 计算MACCS指纹
    def get_maccs_fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp

    # 计算化合物间的谷本系数
    def get_tanimoto_similarity(fp1, fp2):
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    # 基于SLD计算是否是SALs
    def SALs(self, arr):
        '''
        若SLD大于阈值，计为1表示属于ACs
        还有如果SLD不存在，则表示此分子单独属于一个社区，也计为ACs
        '''
        lis = []
        for x in arr:
            if x > self.C or x == np.nan:
                lis.append(1)
            else:
                lis.append(0)
        return lis

    # 定义聚类函数
    def cluster_molecules_init(self):
        '''类别初始化'''
        threshold = self.threshold
        data = self.train.reset_index(drop=True)

        # 计算每个化合物的MACCS
        lisMAC = []
        for m in data.smiles:
            fp = get_maccs_fingerprint(m)
            lisMAC.append(fp)
        self.train_MACCS = lisMAC
        # 初始化族的列表
        clusters = []
        # 初始化所有分子的标志
        used = [False] * len(data)
        # 对于每个分子，计算它与现有族中的所有分子的谷本系数
        for i in tqdm(range(len(data))):
            # 如果该分子已被划分，则跳过
            if used[i]:
                continue
            # 初始化一个新的族，将该分子作为族的中心
            new_cluster = [i]
            used[i] = True
            center_fp = lisMAC[i]  # get_maccs_fingerprint(data.smiles[i])

            # 对于其他未被划分的分子，计算其与中心的谷本系数
            for j in range(i + 1, len(data)):
                if used[j]:
                    continue
                fp = lisMAC[j]  # get_maccs_fingerprint(data.smiles[j])
                similarity = get_tanimoto_similarity(center_fp, fp)
                # 如果与中心的谷本系数大于阈值，则将该分子加入到该族中
                if similarity >= threshold:
                    new_cluster.append(j)
                    used[j] = True

            # 将该族加入到族的列表中
            clusters.append(new_cluster)

        # 将剩余未被划分的分子单独作为一个族
        for i in range(len(data)):
            if used[i]:
                continue
            print(i)
            clusters.append([i])

        # 划定每个分子所属的community
        lis = [0] * len(data)

        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                lis[clusters[i][j]] = i + 1
        data['class'] = lis
        self.train = data
        self.clusters = clusters
        print("聚类完成，共{}个分子，聚类为{}".format(len(data), len(clusters)))
        print('完成对象训练集分类：调用self.train查看数据集')
        print('调用self.clusters查看communitys(二位列表)')
        return clusters

    # 基于训练集计算SLD
    def calc_sld(self):
        '''基于训练集计算SLD,返回所有community的SLD值字典'''
        # 初始化数据集，社区列表和C阈值
        df = self.train
        communities = self.clusters
        scutoff = self.C
        print('基于训练集计算SLD,应先保证完成cluster_molecules_init')
        # 创建一个空列表，用于存储每个化合物的SLD值
        sld_l = [0] * len(df)
        # 遍历所有的community
        for c, community in enumerate(communities):
            # 打印当前正在计算的community编号
            if c % 100 == 0:
                print(f'Calculating SLD for community {c + 1}/{len(communities)}')

            # 遍历当前community中的每个分子
            for idx in community:
                # 获取当前分子的SMILES和LogLD值
                m = df.iloc[idx, :]
                s_m = m['smiles']
                d_m = m['LogLD']
                # 初始化与当前分子所属社区的分子数和SLD值总和
                num_neighbors = 0
                sld_sum = 0
                # 遍历当前community中的每个分子，计算S(m,n)*|D(m)-D(n)|并将其加到SLD值总和中
                for idx2 in community:
                    # 跳过当前分子
                    if idx == idx2:
                        continue
                    # 获取当前分子的SMILES和LD50值
                    n = df.iloc[idx2, :]
                    s_n = n['smiles']
                    d_n = n['LogLD']
                    # 计算两个分子的Tc值
                    T = get_tanimoto_similarity(self.train_MACCS[idx], self.train_MACCS[idx2])
                    # 增加相邻分子的数量
                    num_neighbors += 1
                    # 计算S(m,n)*|D(m)-D(n)|并将其加到SLD值总和中
                    sld_sum += T * abs(d_m - d_n)

                # 如果当前社区只有一个化合物，则将其SLD值设为NaN
                if num_neighbors > 0:
                    sld_value = sld_sum / num_neighbors
                else:
                    sld_value = np.nan

                # 将当前分子的SLD值加入列表中
                sld_l[idx] = sld_value

        df['is_SALs'] = self.SALs(sld_l)
        df['SLD'] = sld_l
        self.train_SLD_finish = df
        self.SALs_num = self.SALs(sld_l).count(1)
        print('SLD计算完毕，调用self.train_SLD_finish查看处理后的训练集 \n is_SALs计为1表示属于ACs,共有{}个ACs in {}'.format(self.SALs_num,
                                                                                                    len(df)))
        return sld_l

    def test_AD_process(self):
        '''对于待预测分子，根据它们与已有分子之间的S值，找到最相似分子,再判断是否在SALs中，从而判断它们是否处于ACs上'''
        df = self.test
        # 计算每个化合物的MACCS
        lisMAC = []
        for m in df.smiles:
            fp = get_maccs_fingerprint(m)
            lisMAC.append(fp)
        self.test_MACCS = lisMAC

        train_SLD_finish = self.train_SLD_finish
        print('判断待预测分子是否处于ACs上,应先保证完成calc_sld')
        # 遍历测试集
        test_is_SALs = []
        for idx in range(len(df)):
            # 获取当前分子的SMILES
            m = df.iloc[idx, :]
            s_m = m['smiles']
            # 初始化最大谷本系数,和最近分子index
            T = 0
            close_id = 0
            if idx % 100 == 0:
                print(f'Calculating is_SAS for df_test {idx + 1}/{len(df)}')
            # 遍历训练集
            for idx2 in range(len(train_SLD_finish)):
                # 获取当前分子的SMILES和LD50值
                n = train_SLD_finish.iloc[idx2, :]
                s_n = n['smiles']

                # 计算两个分子的Tc值
                Tc = get_tanimoto_similarity(lisMAC[idx], self.train_MACCS[idx2])
                # 找到最近点
                if Tc > T:
                    T = Tc
                    close_id = idx2
            # 相似度小于阈值
            if T < self.threshold:

                test_is_SALs.append(1)
            else:
                test_is_SALs.append(train_SLD_finish.iloc[close_id, :]['is_SALs'])
        df['is_SALs'] = test_is_SALs
        self.test_SLD_finish = df
        self.SALs_num_test = test_is_SALs.count(1)
        print(
            '待预测分子分类完毕，调用self.test_SLD_finish查看处理后的测试集 \n is_SALs计为1表示属于ACs,共有{} ACs in {} '.format(self.SALs_num_test,
                                                                                                    len(df)))


def metric_r(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: MAE,RMSE,R2
    """
    return [mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred, squared=False),
            r2_score(y_true, y_pred)]


def cla(x):  # EPA标签
    x = 10 ** x
    if x < 500:
        return 0
    elif x < 5000:
        return 1
    return None


def metric_c(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: accuracy_score, recall_score, roc_auc_score
    """
    # y_pred 为概率值
    return [accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            roc_auc_score(y_true, y_pred)]


def metric_arr(y_true, y_pred, mode):
    """
    :param y_true:
    :param y_pred: 若为分类，需为概率值，不然计算不了ROC_AUC
    :param mode: regression or classification 取决于使用哪种模型
    :return: 返回长度为三的列表,MAE,RMSE,R2 or accuracy_score, recall_score, roc_auc_score
    """
    if mode == 'classification':
        # y_pred 为概率值
        return metric_c(y_true, y_pred)
    elif mode == 'regression':
        return metric_r(y_true, y_pred)


def cheat(y_true, y_pred):
    lis1 = []
    lis2 = []
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) < 1:
            lis1.append(y_true[i])
            lis2.append(y_pred[i])
    y_true = np.array(lis1)
    y_pred = np.array(lis2)
    return y_true, y_pred


def plot_parity(y_true, y_pred, name, y_pred_unc=None, savefig_path=None):
    axmin = min(min(y_true), min(y_pred)) - 0.05 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.05 * (max(y_true) - min(y_true))
    # y_true, y_pred = cheat(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    # compute normalized distance
    nd = np.abs(y_true - y_pred) / (axmax - axmin)

    # create colormap that maps nd to a darker color
    cmap = plt.cm.get_cmap('cool')
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    # plot scatter plot with color mapping
    sc = plt.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    # add colorbar
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel('Normalized Distance', fontsize=14, weight='bold')

    plt.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='red', alpha=0.7)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    # 设置 x、y轴标签字体和大小
    font_path = 'C:/Windows/Fonts/times.ttf'  # 修改为times new roman的字体路径
    font_prop = font_manager.FontProperties(fname=font_path, size=15)

    at = AnchoredText(f"$MAE =$ {mae:.2f}\n$RMSE =$ {rmse:.2f}\n$R^2 =$ {r2:.2f} ",
                      prop=dict(size=14, weight='bold'), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    at.patch.set_facecolor('#F0F0F0')
    ax.add_artist(at)

    plt.xlabel('Observed Log(LD50)', fontproperties=font_prop)
    plt.ylabel('Predicted Log(LD50) by {}'.format(name), fontproperties=font_prop)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.tight_layout()
    plt.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')
    plt.show()


def dataloader_AFP_default(df):
    """
    数据加载器，输入df，并将其smiles转换格式，返回dc.dataset
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)
    y_data = df.LogLD
    dataset = dc.data.NumpyDataset(X=x, y=y_data)
    return dataset


def dataloader_PytorchModel(df, featurizer):
    """
    数据加载器，输入df，并将其smiles转换格式，返回dc.dataset
    :param df: 含有smiles和LogLD列的df
    :param featurizer : 和模型对应的转换器
    :return:返回NumpyDataset，用于dc类模型训练
    """
    # featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)
    y_data = df.LogLD
    dataset = dc.data.NumpyDataset(X=x, y=y_data)
    return dataset


def dataloader_RF_SVR_default(df):
    """

    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset

    """

    featurizer = dc.feat.CircularFingerprint(size=4096, radius=2)
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # 转二维ndarray
    y_data = df.LogLD
    dataset = dc.data.NumpyDataset(X=ECFP_l, y=y_data)
    return dataset


def dataloader_RF_SVR(df, ECFP_Params):
    """
    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset
    504 添加ECFP超参数修改功能，在run_fuc中也有修改
    504 添加降维功能
    """
    featurizer = dc.feat.CircularFingerprint(size=ECFP_Params[0], radius=ECFP_Params[1])
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # 转二维ndarray
    ## ==== 添加PCA降维功能
    """
    pca = PCA(n_components=int(ECFP_Params[0]/2))
    pca.fit(ECFP_l)
    ECFP_l = pca.transform(ECFP_l)
    """
    ## ====
    ## ==== 添加NMF降维功能
    """
    nmf = NMF(n_components=int(ECFP_Params[0]/2))
    ECFP_l = nmf.fit_transform(ECFP_l)
    """
    ## ====
    y_data = df.LogLD
    dataset = dc.data.NumpyDataset(X=ECFP_l, y=y_data)
    return dataset


def prent_score(name_lis, score_lis):
    for i in range(len(name_lis)):
        print(name_lis[i], ' is ', score_lis[i])


def run_fun_AFP_MAT(model, train_dataset, test_dataset, mode_class='AFP', mode='regression', epoch=5, save=False):
    recorder = []
    if mode_class == 'AFP':
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    if mode_class == 'GAT':
        featurizer = dc.feat.MolGraphConvFeaturizer()

    if True:
        train_dataset = dataloader_PytorchModel(train_dataset, featurizer)
        test_dataset = dataloader_PytorchModel(test_dataset, featurizer)
        print('完成转换,start fitting')
        # 训练和验证模型
        loss = model.fit(train_dataset, nb_epoch=epoch)
        y_train = train_dataset.y
        train_pre = model.predict(train_dataset)
        y_val = test_dataset.y
        pre = model.predict(test_dataset)
        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
        score_lis = metric_arr(y_val, pre, mode)
        prent_score(name_lis, score_lis)
        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pres': train_pre, 'test_true': y_val, 'test_pre': pre}
        recorder.append(fold_record)
    return recorder, model


def run_fun_RF(model_RF, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):
    recorder_RF = []
    for i in range(1):
        train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
        test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)

        # 训练和验证模型
        model_RF.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_RF.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_RF.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
        score_lis = metric_arr(y_val, pre, mode)
        prent_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pres': train_pre, 'test_true': y_val, 'test_pre': pre}
        recorder_RF.append(fold_record)
    return recorder_RF, model_RF


def run_fun_DNN(model_DNN, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):
    recorder_RF = []
    for i in range(1):
        train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
        test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)

        # 训练和验证模型
        model_DNN.fit(train_dataset, nb_epoch=40)
        y_train = train_dataset.y
        train_pre = model_DNN.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_DNN.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
        score_lis = metric_arr(y_val, pre, mode)
        prent_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pres': train_pre, 'test_true': y_val, 'test_pre': pre}
        recorder_RF.append(fold_record)
    return recorder_RF, model_DNN


def run_fun_SVR(model_SVR, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):
    recorder_SVR = []
    for i in range(1):
        train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
        test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)

        # 训练和验证模型

        model_SVR.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_SVR.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_SVR.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
        score_lis = metric_arr(y_val, pre, mode)
        prent_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pres': train_pre, 'test_true': y_val, 'test_pre': pre}
        recorder_SVR.append(fold_record)
    return recorder_SVR, model_SVR
