import warnings

warnings.filterwarnings("ignore")
import numpy as np
import chemprop
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import svm
import pickle
from deepchem.models.torch_models import AttentiveFPModel
from utils import plot_parity, run_fun_SVR, run_fun_RP, run_fun_AFP, dataloader_RF_SVR, dataloader_AFP, ADFP_AC
from sklearn.linear_model import LinearRegression

class Args:
    def __init__(self, train, test, predict_df=None, AFP=False, RF=False, MPNN=False, SVR=False, save_r=False,
                 plot=False, AD_FP=False,
                 S_C=[0.8, 0.4]):
        self.train = train
        self.test = test
        self.model_lis = [AFP, RF, MPNN, SVR]
        self.save_r = save_r
        self.AD_FP = AD_FP
        self.S_C = S_C
        self.plot = plot
        self.predict_df = predict_df


class ensemble_models:
    def __init__(self):
        self.recorder_AFP = None
        self.arg = None
        self.recorder_MPNN = None
        self.recorder_RF = None
        self.recorder_SVR = None

    def Lay1_models_train(self, Args, save=False):
        """
        训练不同的模型并生成它们的预测结果。如果plot为True，还会生成预测值与实际值的对比图。
        AD_FP:控制是否进行适用域处理,通过S_C列表输入阈值S和C
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

        # 输入数据
        print('train is{} ,test is {}'.format(train.shape, test.shape))

        # 初始化输出
        out_recorder = {}
        # AFP模型
        # 定义模型
        if AFP:
            print('start training AFP ')
            model_AFP = AttentiveFPModel(n_tasks=1,
                                         batch_size=32, learning_rate=0.001
                                         )

            recorder_AFP, model_AFP = run_fun_AFP(model_AFP, train_dataset=train, test_dataset=test, epoch=40)
            if plot:
                plot_parity(recorder_AFP[0]['test_true'].reshape(-1), recorder_AFP[0]['test_pre'].reshape(-1), 'AFP')
            out_recorder['AFP'] = recorder_AFP[0]['test_pre'].reshape(-1)
            if save:
                model_AFP.save_checkpoint(model_dir='../models/tmp/AFP')
            self.recorder_AFP = recorder_AFP
            print('============AFP over============')
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
            out_recorder['MPNN'] = np.array(preds).reshape(-1)
            self.recorder_MPNN = recorder_MPNN
            print('============MPNN over============')
        if RF:
            print('start training RF ')
            model_RF = dc.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                              model_dir='/models/tmp/RF')
            recorder_RF, model_RF = run_fun_RP(model_RF, train_dataset=train, test_dataset=test)
            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), recorder_RF[0]['test_pre'].reshape(-1), 'RF')
            out_recorder['RF'] = recorder_RF[0]['test_pre'].reshape(-1)
            if save:
                model_RF.save()
            self.recorder_RF = recorder_RF
            print('============RF over============')
        # SVR
        if SVR:
            print('start training SVR ')
            model_SVR = dc.models.SklearnModel(svm.SVR(C=1),
                                               model_dir='/models/tmp/SVR')
            recorder_SVR, model_SVR = run_fun_SVR(model_SVR, train_dataset=train, test_dataset=test)

            if plot:
                plot_parity(np.array(test.LogLD).reshape(-1), recorder_SVR[0]['test_pre'].reshape(-1), 'SVR')
            out_recorder['SVR'] = recorder_SVR[0]['test_pre'].reshape(-1)
            if save:
                model_SVR.save()
            self.recorder_SVR = recorder_SVR
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
                file = open('../models/tmp/ADFP/ADer.pickle')
                pickle.dump(AD_FP, file)
                file.close()



        if AFP and MPNN and RF and SVR:
            dic = {'AFP': recorder_AFP[0]['test_pre'].reshape(-1),
                   'MPNN': np.array(preds).reshape(-1),
                   'SVR': recorder_SVR[0]['test_pre'].reshape(-1),
                   'RF': recorder_RF[0]['test_pre'].reshape(-1),
                   'true': recorder_AFP[0]['test_true'].reshape(-1),
                   }
            df_test = pd.DataFrame(dic)
            if AD_FP:
                df_test['out'] = is_SALs
                df_test = df_test[df_test['out'] == 0]
            if save_r:
                df_test.to_csv('tmp/Lay1_test.csv')
            # 保存测试集以及其模型预测情况
            self.test_df = df_test
            dic2 = {'AFP': recorder_AFP[0]['train_pres'].reshape(-1),
                    'MPNN': np.array(preds_train).reshape(-1),
                    'SVR': recorder_SVR[0]['train_pres'].reshape(-1),
                    'RF': recorder_RF[0]['train_pres'].reshape(-1),
                    'true': recorder_AFP[0]['train_true'].reshape(-1),
                    }
            df_train = pd.DataFrame(dic2)
            if save_r:
                df_train.to_csv('tmp/Lay1_predict.csv')
            # 保存测试集以及其模型预测情况 
            self.train_df = df_train
            return df_test, df_train
        else:
            return out_recorder
        return None

    def Lay1_models_predict(self, Args):
        predict_df = Args.predict_df
        [AFP, RF, MPNN, SVR] = Args.model_lis
        if AFP:
            model_AFP = AttentiveFPModel(n_tasks=1,
                                         batch_size=16, learning_rate=0.001,
                                         )
            model_AFP.restore(model_dir='../models/tmp/AFP')
            pre_AFP = model_AFP.predict(dataloader_AFP(predict_df))

        if RF:
            model_RF = dc.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                              model_dir='/models/tmp/RF')

            model_RF.reload()
            pre_RF = model_RF.predict(dataloader_RF_SVR(predict_df))
        if SVR:
            model_SVR = dc.models.SklearnModel(svm.SVR(C=1),
                                               model_dir='/models/tmp/SVR')
            model_SVR.reload()
            pre_SVR = model_SVR.predict(dataloader_RF_SVR(predict_df))
        if MPNN:
            predict_df[['smiles', 'LogLD']].to_csv('tmp/MPNN/predict_df.csv', index=False)
            arguments = [
                '--test_path', 'tmp/MPNN/predict_df.csv',
                '--preds_path', 'tmp/MPNN/predict_df2.csv',
                '--checkpoint_dir', 'tmp/MPNN/test_checkpoints_reg',
            ]
            args = chemprop.args.PredictArgs().parse_args(arguments)
            pred_MPNN = chemprop.train.make_predictions(args=args)

        if AFP and MPNN and RF and SVR:
            dic = {'AFP': pre_AFP.reshape(-1),
                   'MPNN': np.array(pred_MPNN).reshape(-1),
                   'SVR': pre_SVR.reshape(-1),
                   'RF': pre_RF.reshape(-1),
                   'true': np.array(predict_df['LogLD']).reshape(-1),
                   }
            df_pre = pd.DataFrame(dic)
            if Args.AD_FP:
                with open('../models/tmp/ADFP/ADer.pickle', 'rb') as file:
                    # 从文件中数据处理器
                    ADer = pickle.load(file)
                    ADer.test = predict_df
                ADer.test_AD_process()
                df_test_sal = ADer.test_SLD_finish.copy()

                # 除杂
                l = ['C', '[C]', '[Na+].[Br-]', '[S]']
                df_test_sal = df_test_sal[~df_test_sal['smiles'].isin(l)]
                is_SALs = df_test_sal.is_SALs
                df_pre['out'] = is_SALs
                df_pre = df_pre[df_pre['out'] == 0]

            if Args.save_r:
                df_pre.to_csv('tmp/predict_df.csv')

            return df_pre


    def Lay2_models_train(self,Args):
        pass

def Lay2_models(meta_data_train, meta_data_test,
                i=0, MLR=False, RF=False, SVR=False, save_r=False, plot=False):
    '''训练不同的模型并生成它们的预测结果。如果plot为True，还会生成预测值与实际值的对比图。'''
    # 将"TRUE"列作为响应变量，前四列作为解释变量
    X = meta_data_train[['MPNN', 'AFP', 'SVR', 'RF']]
    y = meta_data_train['true']

    Xt = meta_data_test[['MPNN', 'AFP', 'SVR', 'RF']]
    yt = meta_data_test['true']

    recorder = []
    # 拟合线性回归模型
    if MLR:
        model_MLR = LinearRegression().fit(X, y)
        p_MLR = model_MLR.predict(X)
        pres_MLR = model_MLR.predict(Xt)
        if plot:
            plot_parity(yt, pres_MLR, 'MLR_test')
    if RF:
        model_RF = RandomForestRegressor().fit(X, y)
        p_RF = model_RF.predict(X)
        pres_RF = model_RF.predict(Xt)
        if plot:
            plot_parity(yt, pres_RF, 'RF_test')
    if SVR:
        model_SVR = svm.SVR().fit(X, y)
        p_SVR = model_SVR.predict(X)
        pres_SVR = model_SVR.predict(Xt)
        if plot:
            plot_parity(yt, pres_SVR, 'SVR_test')

    if MLR and RF and SVR:
        dic = {'MLR': pres_MLR.reshape(-1),
               'SVR': pres_SVR.reshape(-1),
               'RF': pres_RF.reshape(-1),
               'true': yt,
               }
        df_test = pd.DataFrame(dic)

        if save_r:
            df_test.to_csv('Lay2_test_fold{}.csv'.format(i))

        dic2 = {'MLR': p_MLR.reshape(-1),
                'SVR': p_SVR.reshape(-1),
                'RF': p_RF.reshape(-1),
                'true': y,
                }
        df_train = pd.DataFrame(dic)

        if save_r:
            df_test.to_csv('Lay2_test_fold{}.csv'.format(i))
        if save_r:
            df_train.to_csv('Lay2_train_fold{}.csv'.format(i))
    return df_train, df_test

