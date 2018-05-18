import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
from tqdm import tqdm
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def preprocess(data: pd.DataFrame):
    """ 对数据进行预处理
    """

    def fill_outliers(col: pd.Series):
        """ Remove outliers of each col
        """
        mean = col.mean()
        std = col.std()
        upper = mean + 3 * std
        lower = mean - 3 * std
        col[col > upper] = np.floor(upper)
        col[col < lower] = np.floor(lower)
        return col.values

    # 处理离散值 & 填充空值(使用众数填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode).astype('float64')

    return data


with timer("split train and test dataset!!!"):
    # 读取训练集和测试集
    X_train = pd.read_csv('./dataset/atec_anti_fraud_train.csv', encoding='utf-8',
                          low_memory=False, parse_dates=['date'],index_col='id')
    X_test = pd.read_csv('./dataset/atec_anti_fraud_test_a.csv', encoding='utf-8',
                         low_memory=False, parse_dates=['date'],index_col='id')

    col_train_num, col_test_num = X_train.columns, X_test.columns

    X_train, X_test = X_train[col_train_num], X_test[col_test_num]

    X_train = X_train.drop(columns=['label','date'])

    X_test=X_test.drop(columns=['date'])

    print(X_train.shape, X_test.shape)
    print("Start filter features!!!")
    # 筛选缺失率小于0.9的特征
    col_train, col_test = [], []
    for item in X_train.columns:
        tmp = np.sum(X_train[item].isnull()) / len(X_train)
        if tmp < 0.5:
            col_train.append(item)
    for item in X_test.columns:
        tmp = np.sum(X_test[item].isnull()) / len(X_test)
        if tmp < 0.5:
            col_test.append(item)
    # 选择训练集和测试集的交集
    col = [item for item in col_train if item in col_test]
    print('len(col):', len(col))
    X_train, X_test = X_train[col], X_test[col]
    X_train, X_test = preprocess(X_train), preprocess(X_test)
    print(X_train.shape, X_test.shape)
    X_train.to_csv("./dataset/x_train.csv", encoding='utf-8')
    X_test.to_csv("./dataset/x_test.csv", encoding='utf-8')
    del X_train,X_test
    gc.collect()
"""
X_train, X_test = X_train.fillna(0), X_test.fillna(0)
# print(X_train.info(), X_test.info())

# PCA
X_train = PCA(n_components=15).fit_transform(X_train)
X_test = PCA(n_components=15).fit_transform(X_test)

print(X_train.shape, X_test.shape)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
print(X_train.info(), X_test.info())

X_train.to_csv("./dataset/x_train.csv", encoding='utf-8')
X_test.to_csv("./dataset/x_test.csv", encoding='utf-8')
"""
