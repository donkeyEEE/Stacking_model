import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,recall_score

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredText
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import gaussian_kde
from utils import plot_parity, metric_r


def recorder_read(path, Lay_nb):
    """
    :param path: 路径
    :param Lay_nb: 读取第几层模型
    :return: 返回k*2的二维列表，k为fold数，2为训练集和测试集结果。其值为df格式，包含true值和模型预测结果
    """
    recorder = []
    for i in range(5):
        df_train = pd.read_csv('{}/df_train_L{}_fold{}'
                               .format(path,Lay_nb,i),
                               index_col=0)
        df_test = pd.read_csv('{}/df_test_L{}_fold{}'
                               .format(path,Lay_nb,i),
                               index_col=0)
        recorder.append([df_train,df_test])
    return recorder


def recorder_cal(recorder):
    """
    # 根据recorder计算每个df的指标
    :param recorder: k*2的二维列表，k为fold数，2为训练集和测试集结果。其值为df格式，包含true值和模型预测结果
    :return: 5*2的二维列表，值为df格式，其中为metric_r计算出来的指标
    """
    metric_recorder = []
    for i in range(5):
        lis = []
        for j in range(2):
            dic = {}
            for x in list(recorder[0][0].columns)[1:]:
                arr_x = np.array(recorder[i][j][x])
                arr_x_true = np.array(recorder[i][j]['true'])
                mae, rmse, r2 = metric_r(arr_x_true, arr_x)
                dic[x] = [mae, rmse, r2]
            lis.append(pd.DataFrame(dic, index=['MAE', 'RMSE', 'r2']))
        metric_recorder.append(lis)
    return metric_recorder


def recorder_pre(recorder_metric):
    """
    计算模型五折测试评估指标的平均值
    :param recorder_metric: 5*2的二维列表，其中为metric_r计算出来的指标
    :return:模型性能五折平均值的df
    """
    lis = 0
    for i in range(5):
        lis += recorder_metric[i][1].values
    lis = lis/5
    output_df = pd.DataFrame(lis, index=recorder_metric[i][1].index, columns=recorder_metric[i][1].columns)
    print('模型性能五折平均值')
    return output_df


def recoder_sta(recorder_metric, metrics='RMSE'):
    """
    :param recorder_metric: 5*2的二维列表，值为df格式，其中为metric_r计算出来的指标
    :return: 二维列表，index为模型，Columns为指定指标的五次试验结果
    """
    static_data = []
    for x in list(recorder_metric[0][0].columns):
        lis = [recorder_metric[i][1][x][metrics] for i in range(5)]
        static_data.append(lis)
    return static_data


def plot_bar(static_data_all, name_lis=[]):
    # 设置全局字体为Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.ylim((0.48,0.67))
    # 指定颜色列表
    colors = [(229, 190, 121), (62, 134, 181), (51, 100, 133), (149, 167, 126)]

    # 转换为 Matplotlib 可接受的颜色格式
    colors = [(r/255, g/255, b/255) for (r, g, b) in colors]

    # 创建箱线图
    sns.set(style="white", font='Times New Roman', palette=colors,font_scale=1.6)
    #sns.set(style="darkgrid", font='Times New Roman', palette=colors)
    ax = sns.boxplot(data=static_data_all,width=0.7)

    # 添加标签和标题
    if len(name_lis) != 0:
      ax.set_xticklabels(name_lis,
                       fontname='Times New Roman', fontsize=15, rotation = 30)

    #ax.set_yticks([0.400,0.450,0.500,0.550],fontsize=100)
    #ax.set_ylabel('$RMSE$', fontname='Times New Roman', fontsize=20, rotation=0)

    #ax.set_yticklabels(['0','0.4','0.5','0.6','0.7','0.8'], fontdict={'fontname': 'Times New Roman', 'fontsize': 20})

    ax.yaxis.set_label_coords(-0.2, 0.5)
    plt.subplots_adjust(left=0.2)
    # plt.savefig('Five-fold all',dpi=600)