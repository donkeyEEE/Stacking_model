import pandas as pd
from sklearn.model_selection import KFold
from datetime import datetime


class db:
    def __init__(self, file_path=None):
        self.fp = file_path
        if file_path:
            self.data = self.read_data(self.fp)

    @staticmethod
    def read_data(file_path):
        df = pd.read_csv(file_path, index_col=0, dtype={2: float}).reset_index(drop=True)
        # df = df.groupby(['smiles']).mean().reset_index()
        df = df.reindex(columns=['CASRN', 'LogLD', 'smiles'])
        print('data already finished size ={}'.format(df.shape[0]))
        return df

    def get_data(self):
        self.fp = "E:\学习\文献库\pythonProject\DataBase\DB.csv"
        self.data = self.read_data(self.fp)

    def search(self, x):
        df = self.data
        location = df.index[df.isin(x).any(axis=1)]
        subdf = df.loc[location]
        return [location, subdf]

    @staticmethod
    def merge_and_average(a, b):
        # 对a的第二列进行修改
        print(a.shape)
        print(b.shape)
        b = b.set_index(a.index)

        a.iloc[:, 1] = (a.iloc[:, 1] + b.iloc[:, 1]) / 2.0

        # 将合并后的数据添加到a中，并返回
        return a

    def add_fuc(self, new_df, how="average"):
        # 检查输入是否为三列的数据框
        if len(new_df.columns) != 3:
            print("Error: Input dataframe should have 3 columns")
            return None

        # 检查输入数据框的smiles列中是否有与数据集相同的元素
        matching_rows = self.data.iloc[:, 2].isin(new_df.iloc[0:, 2])
        matching_rows2 = new_df.iloc[:, 2].isin(self.data.iloc[0:, 2])
        matching_df = self.data[matching_rows]
        self.data = self.data[~matching_rows]

        overlap = len(matching_df)

        if overlap > 0:
            print(f"{overlap} records with overlapping values found in dataset")

            # 先删除重合元素

            matching_df2 = new_df[matching_rows2]
            new_df = new_df[~matching_rows2]

            # 将融合后的记录添加到数据集中
            self.data.update(new_df)

            # 合并匹配的记录和新的记录
            ddf = self.merge_and_average(matching_df, matching_df2)
            print(ddf)
            self.data = pd.concat([self.data, ddf])

            # 打印添加记录的数量
            print(f"{len(new_df)} records added to dataset")
            print(f"{len(matching_df)} records refresh to dataset")
        else:
            # 将新的记录添加到数据集中
            self.data = pd.concat([self.data, new_df])
            print(f"{len(new_df)} records added to dataset")
        self.data = self.data.reset_index(drop=True)

    def get_folds(self, df_data_p=None, fold=5, save_splits=False):
        """折叠切分数据集函数,返回迭代器
           df_data_p可以外接输入，或者默认加载过的数据
           默认5折cv
           save_splits 是否保存切割结果
        """
        df_data_p = self.data  # 得先加载数据再进行切割
        kf = KFold(n_splits=fold, shuffle=True, random_state=20)  # 设置折数
        for i, (train_index, test_index) in enumerate(kf.split(df_data_p)):
            print('fold{}'.format(i))
            train = df_data_p.iloc[train_index]
            test = df_data_p.iloc[test_index]
            if save_splits:
                train.to_csv('train_fold{}.csv'.format(i))
                test.to_csv('test_fold{}.csv'.format(i))
            yield (train, test)

    def save_data(self, save_path=None):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime('%d-%H-%M')

        # 将当前时间添加到文件名中
        filename = f"dataframe_refresh_{current_time}.csv"

        self.data.to_csv('../DataBase/{}'.format(filename))
