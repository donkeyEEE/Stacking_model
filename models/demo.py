import numpy as np
import pandas as pd
from Layers import ensemble_models
from DB.DB import db
from utils import Args
from utils import metric_r, metric_c


# 加载数据集
data_loader = db('../DataBase/DB.csv')
data_loader.data = data_loader.data.iloc[:20, :]
# 五折切割
train, test = next(data_loader.get_folds())


# 定义参数类
# MPNN尚不支持py中运行，GAT和DNN不支持保存模型以及对未知数据进行预测
args = Args(train, test, AFP=True, RF=True, MPNN=False,
            SVR=True, save_r=False, plot=True, GAT=False, DNN=False,
            AD_FP=False)
# 实例化集成模型
eb = ensemble_models()

# 训练第一层模型，save选择是否保存模型
eb.Lay1_models_train(args , save=True)


# 设置第二层模型参数 args.model_lis_L2
args.model_lis_L2 = [True for i in range(3)]
eb.Lay2_models_train(args , save=True)

# 训练第三层模型，
# 第三层模型为非负最小二乘，只有保存权重和偏置
eb.Lay3_models_train(args)