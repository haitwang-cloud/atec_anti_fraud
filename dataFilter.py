import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
from tqdm import tqdm
import time
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler


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

    # 处理离散值 & 填充空值(使用均值填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = data[col_name].fillna(data[col_name].mean())
    #标准化    
    return data

with timer("split train and test dataset!!!"):
    # 读取训练集和测试集
    X_train = pd.read_csv('./dataset/atec_anti_fraud_train.csv', encoding='utf-8',
                          low_memory=False, parse_dates=['date'],index_col='id')
    X_test = pd.read_csv('./dataset/atec_anti_fraud_test_b.csv', encoding='utf-8',
                         low_memory=False, parse_dates=['date'],index_col='id')
    col_train_num, col_test_num = X_train.columns, X_test.columns
    X_train, X_test = X_train[col_train_num], X_test[col_test_num]
    X_train_label,X_train_date=X_train.pop('label'),X_train.pop('date')
    X_test_date=X_test.pop('date')
    print(X_train.shape, X_test.shape)
    print("Start filter features!!!")
    # 筛选缺失率小于0.6的特征
    col_train, col_test = [], []
    for item in X_train.columns:
        tmp = np.sum(X_train[item].isnull()) / len(X_train)
        if tmp < 1:
            col_train.append(item)
    for item in X_test.columns:
        tmp = np.sum(X_test[item].isnull()) / len(X_test)
        if tmp <1:
            col_test.append(item)
    # 选择训练集和测试集的交集
    col = [item for item in col_train if item in col_test]
    print('len(col):', len(col))
    X_train, X_test = X_train[col], X_test[col]
    X_train, X_test = preprocess(X_train), preprocess(X_test)
    X_train, X_test = pd.DataFrame(X_train),pd.DataFrame(X_test)
    
    X_train=pd.concat([X_train_label,X_train_date,X_train],axis=1)
    X_test=pd.concat([X_test_date,X_test],axis=1)

    X_train_col,X_test_col=col.copy(),col.copy()   
    X_train_col.insert(0,'label')
    X_train_col.insert(1,'date')
    X_test_col.insert(0,'date')

    print(X_train.shape, X_test.shape)
    print("Start writing")
    X_train.to_csv("./dataset/x_train.csv", encoding='utf-8',header=X_train_col)
    X_test.to_csv("./dataset/x_test_b.csv", encoding='utf-8',header=X_test_col)
    del X_train,X_test
    gc.collect()

