import pandas as pd
from LAY2 import ensemble_models
from DB.DB import db
from utils import Args

# 加载数据集
data_loader = db('../DataBase/DB2.csv')
# data_loader.get_data()
data_loader.data = data_loader.data.iloc[:10, :]
# 五折切割
train, test = next(data_loader.get_folds())

# 定义参数类
args = Args(train, test, predict_df=test, MPNN=True, save_r=False, plot=False,
            AD_FP=False)

# 实例化集成模型
eb = ensemble_models()

eb.Lay1_models_predict(args)

print(eb.L1_predict_df)
