import warnings

warnings.filterwarnings("ignore")
import numpy as np
import chemprop
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import svm
import pickle
from deepchem.models.torch_models import AttentiveFPModel, MATModel, GATModel
from utils import plot_parity, run_fun_SVR, run_fun_RF, run_fun_AFP_MAT, dataloader_RF_SVR, dataloader_PytorchModel, \
    ADFP_AC
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import torch

class ensemble_models:
    def __init__(self):
        self.arg = None
        # 保存每个基模型的训练/测试数据
        self.recorder_AFP = None
        self.recorder_MPNN = None
        self.recorder_RF = None
        self.recorder_SVR = None
        # 保存预测化合物列表，并且保存结果
        self.predict_df = None
        # 保存所有第一层使用的模型的预测值和真实值
        self.L1_train_df = None
        self.L1_test_df = None
        self.L1_predict_df = None

        # 保存所有第二层使用的模型的预测值和真实值
        self.L2_train_df = None
        self.L2_test_df = None
        self.L2_predict_df = None

        # 保存第三层训练结果和测试结果
        self.L3_test_df = None
        self.L3_train_df = None

        # 保存预测结果
        self.L3_predict_array = None

        # 第三层NNLS的权重
        self.lis_coef = np.array([0.2, 0.5, 0.3])
        self.lis_res = 0.232

    def Lay1_models_train(self, Args, save=False):
        """
        训练不同的模型并生成它们的预测结果。
        如果plot为True，还会生成测试集的预测值与实际值的对比图。
        """

        train = Args.train
        test = Args.test
        [AFP, RF, MPNN, SVR] = Args.model_lis
        save_r = Args.save_r
        plot = Args.plot
        AD_FP = Args.AD_FP
        S_C = Args.S_C

        # 除杂
        l = ['C', '[C]', '[Na+].[Br-]', '[S]']
        train = train[~train['smiles'].isin(l)]
        test = test[~test['smiles'].isin(l)]
        Args.train = train

        self.arg = Args
        # 汇总真实值和使用的模型的训练以及测试结果
        dic_train = {}
        dic_test = {}
        # 记录真实值
        dic_train['true'] = np.array(train.LogLD).reshape(-1)
        dic_test['true'] = np.array(test.LogLD).reshape(-1)

        # 输入数据
        print('size of train is{} ,test is {}'.format(train.shape, test.shape))
        # AFP模型
        # 定义模型
        if AFP:
            print('start training AFP ')
            model_AFP = AttentiveFPModel(mode='regression', n_tasks=1,
                                         batch_size=32, learning_rate=0.001
                                         )

            recorder_AFP, model_AFP = run_fun_AFP_MAT(model_AFP, train_dataset=train, test_dataset=test, epoch=40)
            if plot:
                plot_parity(recorder_AFP[0]['test_true'].reshape(-1), recorder_AFP[0]['test_pre'].reshape(-1), 'AFP')

            if save:
                model_AFP.save_checkpoint(model_dir='tmp/AFP')
            self.recorder_AFP = recorder_AFP
            dic_train['AFP'] = recorder_AFP[0]['train_pres'].reshape(-1)
            dic_test['AFP'] = recorder_AFP[0]['test_pre'].reshape(-1)
            print('============AFP over============')
        # MATModel
        GAT = Args.GAT
        if GAT:
            print('start training GAT ')
            """
            model_MAT = MATModel(mode='regression', n_tasks=1,
                                 batch_size=32, learning_rate=0.001
                                 )
            """
            model_MAT = GATModel(mode='regression', n_tasks=1,
                                 batch_size=32, learning_rate=0.001
                                 )

            recorder_MAT, model_MAT = run_fun_AFP_MAT(model_MAT, mode_class='GAT', train_dataset=train, test_dataset=test, epoch=40)
            if plot:
                plot_parity(recorder_MAT[0]['test_true'].reshape(-1), recorder_MAT[0]['test_pre'].reshape(-1), 'GAT')

            if save:
                model_MAT.save_checkpoint(model_dir='tmp/GAT')
            self.recorder_AFP = recorder_MAT
            dic_train['GAT'] = recorder_MAT[0]['train_pres'].reshape(-1)
            dic_test['GAT'] = recorder_MAT[0]['test_pre'].reshape(-1)
            print('============GAT over============')
        # MPNN
        if MPNN:
            print('start training MPNN ')
            train[['smiles', 'LogLD']].to_csv('tmp/MPNN/train.csv', index=False)
            test[['smiles', 'LogLD']].to_csv('tmp/MPNN/test.csv', index=False)
            recorder_MPNN = {'train_true': train.LogLD, 'test_true': test.LogLD}
            arguments = [
                '--data_path', 'tmp/MPNN/train.csv',
                '--separate_test_path', 'tmp/MPNN/test.csv',
                '--separate_val_path', 'tmp/MPNN/test.csv',
                '--dataset_type', 'regression',
                '--save_dir', 'tmp/MPNN/test_checkpoints_reg',
                '--epochs', '25',  #
                '--num_folds', '1',
                '--ffn_num_layers', '3',
            ]

            args = chemprop.args.TrainArgs().parse_args(arguments)
            mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

            arguments = [
                '--test_path', 'tmp/MPNN/test.csv',
                '--preds_path', 'tmp/MPNN/test1.csv',
                '--checkpoint_dir', 'tmp/MPNN/test_checkpoints_reg',
            ]

            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args)
            recorder_MPNN['test_pre'] = preds

            arguments = [
                '--test_path', 'tmp/MPNN/train.csv',
                '--preds_path', 'tmp/MPNN/train_MPNN.csv',
                '--checkpoint_dir', 'tmp/MPNN/test_checkpoints_reg',
            ]
            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds_train = chemprop.train.make_predictions(args=args)
            recorder_MPNN['train_pre'] = preds
            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), np.array(preds).reshape(-1), 'MPNN')

            self.recorder_MPNN = recorder_MPNN
            dic_train['MPNN'] = np.array(preds_train).reshape(-1)
            dic_test['MPNN'] = np.array(preds).reshape(-1)
            print('============MPNN over============')

        # DNN 基于dc的TorchModel
        # 505
        DNN = Args.DNN
        if DNN:
            print('start training DNN')
            # 定义模型
            DNN_model = torch.nn.Sequential(
                torch.nn.Linear(Args.ECFP_Params[0], Args.ECFP_Params[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(Args.ECFP_Params[0], Args.ECFP_Params[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(Args.ECFP_Params[0], 1)
            )
            DNN_model = dc.models.TorchModel(DNN_model, loss=dc.models.losses.L2Loss())
            # 训练
            from utils import run_fun_DNN
            if Args.ECFP_Params is not None:
                # print(Args.ECFP_Params)
                recorder_DNN, model_DNN = run_fun_DNN(DNN_model, train_dataset=train, test_dataset=test,
                                                    ECFP_Params=Args.ECFP_Params)
            else:
                recorder_DNN, model_DNN = run_fun_DNN(DNN_model, train_dataset=train, test_dataset=test)

            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), recorder_DNN[0]['test_pre'].reshape(-1), 'DNN')

            if save:
                model_DNN.save()
            self.recorder_DNN = recorder_DNN
            dic_train['DNN'] = recorder_DNN[0]['train_pres'].reshape(-1)
            dic_test['DNN'] = recorder_DNN[0]['test_pre'].reshape(-1)
            print('============DNN over============')

        # RF
        if RF:
            print('start training RF ')
            model_RF = dc.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                              model_dir='E:\学习\文献库\pythonProject\models/tmp/RF')

            if Args.ECFP_Params is not None:
                # print(Args.ECFP_Params)
                recorder_RF, model_RF = run_fun_RF(model_RF, train_dataset=train, test_dataset=test,
                                                   ECFP_Params=Args.ECFP_Params)
            else:
                recorder_RF, model_RF = run_fun_RF(model_RF, train_dataset=train, test_dataset=test)

            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), recorder_RF[0]['test_pre'].reshape(-1), 'RF')

            if save:
                model_RF.save()
            self.recorder_RF = recorder_RF
            dic_train['RF'] = recorder_RF[0]['train_pres'].reshape(-1)
            dic_test['RF'] = recorder_RF[0]['test_pre'].reshape(-1)
            print('============RF over============')
        # SVR
        if SVR:
            print('start training SVR ')
            model_SVR = dc.models.SklearnModel(svm.SVR(C=1),
                                               model_dir='E:\学习\文献库\pythonProject\models/tmp/SVR')
            # 添加修改描述符参数的功能
            if Args.ECFP_Params is not None:
                recorder_SVR, model_SVR = run_fun_RF(model_SVR, train_dataset=train, test_dataset=test,
                                                     ECFP_Params=Args.ECFP_Params)
            else:
                recorder_SVR, model_SVR = run_fun_RF(model_SVR, train_dataset=train, test_dataset=test)
            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), recorder_SVR[0]['test_pre'].reshape(-1), 'SVR')

            if save:
                model_SVR.save()
            self.recorder_SVR = recorder_SVR
            dic_train['SVR'] = recorder_SVR[0]['train_pres'].reshape(-1)
            dic_test['SVR'] = recorder_SVR[0]['test_pre'].reshape(-1)
            print('============SVR over============')
        if AD_FP:
            print('开始进行使用域拟合')
            s = S_C[0]
            c = S_C[1]
            ADer = ADFP_AC(train, test, S=s, C=c)
            c1 = ADer.cluster_molecules_init()
            SLD_lis = ADer.calc_sld()
            ADer.test_AD_process()
            df_test_sal = ADer.test_SLD_finish.copy()
            # 除杂
            l = ['C', '[C]', '[Na+].[Br-]', '[S]']
            df_test_sal = df_test_sal[~df_test_sal['smiles'].isin(l)]
            is_SALs = df_test_sal.is_SALs
            if save:
                file = open('tmp/ADFP/ADer.pickle', 'wb')
                pickle.dump(ADer, file)
                file.close()

        df_train = pd.DataFrame(dic_train)
        df_test = pd.DataFrame(dic_test)
        self.L1_train_df = df_train
        self.L1_test_df = df_test

        if Args.save_r:
            df_train.to_csv('tmp/L1_train_data.csv')
            df_test.to_csv('tmp/L1_test_data.csv')

    def Lay1_models_predict(self, Args):
        predict_df = Args.predict_df
        # 除杂
        l = ['C', '[C]', '[Na+].[Br-]', '[S]']
        predict_df = predict_df[~predict_df['smiles'].isin(l)]

        [AFP, RF, MPNN, SVR] = Args.model_lis
        # 保存第一层预测结果
        dic = {}

        if AFP:
            print('AFP predict')
            model_AFP = AttentiveFPModel(n_tasks=1,
                                         batch_size=16, learning_rate=0.001,
                                         )
            model_AFP.restore(model_dir='tmp/AFP')
            pre_AFP = model_AFP.predict(dataloader_PytorchModel(predict_df,
                                                                featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)))
            dic['AFP'] = np.array(pre_AFP).reshape(-1)
        if RF:
            print('RF predict')
            model_RF = dc.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                              model_dir='tmp/RF')

            model_RF.reload()
            pre_RF = model_RF.predict(dataloader_RF_SVR(predict_df, ECFP_Params=[4096,2]))
            dic['RF'] = np.array(pre_RF).reshape(-1)
        if SVR:
            print('SVR predict')
            model_SVR = dc.models.SklearnModel(svm.SVR(C=1),
                                               model_dir='tmp/SVR')
            model_SVR.reload()
            pre_SVR = model_SVR.predict(dataloader_RF_SVR(predict_df, ECFP_Params=[4096, 2]))
            dic['SVR'] = np.array(pre_SVR).reshape(-1)
        if MPNN:
            print('MPNN predict')
            predict_df[['smiles', 'LogLD']].to_csv('tmp/MPNN/predict_df.csv', index=False)
            arguments = [
                '--test_path', 'tmp/MPNN/predict_df.csv',
                '--preds_path', 'tmp/MPNN/predict_df2.csv',
                '--checkpoint_dir', 'tmp/MPNN/test_checkpoints_reg',
            ]
            args = chemprop.args.PredictArgs().parse_args(arguments)
            pred_MPNN = chemprop.train.make_predictions(args=args)
            dic['MPNN'] = np.array(pred_MPNN).reshape(-1)
        df = pd.DataFrame(dic)

        if Args.AD_FP:
            with open('tmp/ADFP/ADer.pickle', 'rb') as file:
                # 从文件中加载数据处理器
                ADer = pickle.load(file)
            ADer.test = predict_df
            ADer.test_AD_process()
            df_test_sal = ADer.test_SLD_finish.copy()
            is_SALs = df_test_sal.is_SALs
            df['out'] = is_SALs
            df = df[df['out'] == 0]

            predict_df['out'] = is_SALs
            predict_df = predict_df[~predict_df['out'] == 0]
            self.predict_df = np.array(predict_df['smiles']).reshape(-1)
            self.L1_predict_df = df.drop(['out'], axis=1)
        else:
            self.predict_df = np.array(predict_df['smiles']).reshape(-1)
            self.L1_predict_df = df

        if Args.save_r:
            df.to_csv('tmp/predict_df.csv')

    def Lay2_models_train(self, Args=None, save=False):
        if not Args:
            Args = self.arg

        [MLR, RF, SVR] = Args.model_lis_L2
        """训练不同的模型并生成它们的预测结果。如果plot为True，还会生成预测值与实际值的对比图。"""
        # 将训练集和测试集划分为特征和目标值
        X = self.L1_train_df[list(self.L1_train_df.columns)[1:]]
        y = self.L1_train_df['true']
        Xt = self.L1_test_df[list(self.L1_test_df.columns)[1:]]
        yt = self.L1_test_df['true']

        dic_train = {'true': np.array(y).reshape(-1)}
        dic_test = {'true': np.array(yt).reshape(-1)}

        def save_models(path, model):
            file = open('tmp/L2/{}.pickle'.format(path), 'wb')
            pickle.dump(model, file)
            file.close()

        print('start fitting L2 models')
        # 拟合线性回归模型
        if MLR:
            print('Start fitting MLP')
            model_MLR = LinearRegression().fit(X, y)
            p_MLR = model_MLR.predict(X)
            pres_MLR = model_MLR.predict(Xt)

            dic_test['MLP'] = pres_MLR
            dic_train['MLP'] = p_MLR

            if Args.plot:
                plot_parity(yt, pres_MLR, 'MLR_test')

            # 保存模型到第二层模型文件夹
            if save:
                save_models('MLP', model_MLR)
                print('Save MLR models')

        # 拟合RF2
        if RF:
            print('Start fitting RF in L2')
            model_RF = RandomForestRegressor().fit(X, y)
            p_RF = model_RF.predict(X)
            pres_RF = model_RF.predict(Xt)

            dic_test['RF'] = pres_RF
            dic_train['RF'] = p_RF

            if Args.plot:
                plot_parity(yt, pres_RF, 'RF_test')
            # 保存模型到第二层模型文件夹
            if save:
                save_models('RF', model_RF)
                print('Save RF models')

        if SVR:
            print('start fitting SVR model')
            model_SVR = svm.SVR().fit(X, y)
            p_SVR = model_SVR.predict(X)
            pres_SVR = model_SVR.predict(Xt)
            dic_test['SVR'] = pres_SVR
            dic_train['SVR'] = p_SVR
            if Args.plot:
                plot_parity(yt, pres_SVR, 'SVR_test')
            # 保存模型到第二层模型文件夹
            if save:
                save_models('SVR', model_SVR)
                print('Save SVR models')

        self.L2_train_df = pd.DataFrame(dic_train)
        self.L2_test_df = pd.DataFrame(dic_test)

        # 保存数据
        if Args.save_r:
            self.L2_train_df.to_csv('tmp/L2_train_data.csv')
            self.L2_test_df.to_csv('tmp/L2_test_data.csv')

    def Lay2_models_predict(self, Args=None):
        # check
        if self.L1_predict_df is None:
            print('Use Lay1_models_predict first')
            return

        # 是否采用默认参数
        if not Args:
            Args = self.arg

        # 获取第一层的数据,划分特征
        # predict_df = self.L1_predict_df
        X = self.L1_predict_df[list(self.L1_predict_df.columns)]

        [MLR, RF, SVR] = Args.model_lis_L2

        # 记录第二层的预测数据
        dic = {}

        # 加载模型
        def load_models(path):
            with open('tmp/L2/{}.pickle'.format(path), 'rb') as file:
                # 从文件中加载数据处理器
                model = pickle.load(file)
                file.close()
            return model

        if MLR:
            model_MLP = load_models('MLP')
            p_MLP = model_MLP.predict(X)
            dic['MLP'] = p_MLP
        if RF:
            model_RF = load_models('RF')
            p_RF = model_RF.predict(X)
            dic['RF'] = p_RF
        if SVR:
            model_SVR = load_models('SVR')
            p_SVR = model_SVR.predict(X)
            dic['SVR'] = p_SVR

        self.L2_predict_df = pd.DataFrame(dic)

    def Lay3_models_train(self, Args=None):
        '''训练NNLS模型并生成它们的预测结果'''
        # 将"TRUE"列作为响应变量，前三列作为解释变量
        X = self.L2_train_df[list(self.L2_train_df.columns)[1:]]
        y = self.L2_train_df['true']

        Xt = self.L2_test_df[list(self.L2_test_df.columns)[1:]]
        yt = self.L2_test_df['true']

        # 拟合NNLS模型
        coefficients, residuals = nnls(X, y)

        # 打印系数和残差
        print('L3,Coefficients:', coefficients)
        print('L3,Residuals:', residuals)

        # 保存第三层权重，相当于保存模型
        self.lis_coef = coefficients
        self.lis_res = residuals

        # 测试集
        test_NNLS = 0
        for i in range(3):
            test_NNLS += coefficients[i] * np.array(Xt.iloc[0:, i])
        self.L3_test_df = pd.DataFrame({'true': np.array(yt).reshape(-1),
                                        'pre': test_NNLS})

        # 训练集
        train_NNLS = 0
        for i in range(3):
            train_NNLS += coefficients[i] * np.array(X.iloc[0:, i])
        self.L3_train_df = pd.DataFrame({'true': np.array(y).reshape(-1),
                                         'pre': train_NNLS})

        if Args.plot:
            plot_parity(yt, test_NNLS, 'L3_test')

    def Lay3_models_predict(self):
        # check
        if self.lis_coef is None:
            print('Lay3_models_train first')

        # 计算输出结果
        out_arr = 0
        for i in range(len(self.lis_coef)):
            k = self.lis_coef[i]
            r = self.lis_res
            out_arr += k * (np.array(self.L2_predict_df.iloc[0:, i]).reshape(-1)) + r

        self.L3_predict_array = out_arr
        self.predict_df = pd.DataFrame({'smiles': self.predict_df, 'pre_LogLD': out_arr})
        print('use L3_predict_array to get output')
