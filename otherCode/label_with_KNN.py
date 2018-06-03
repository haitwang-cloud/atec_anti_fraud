"""
训练集中标签为-1的样本利用KNN对其重新赋值标签为1或0

原始的特征数据不变，只将标记为-1的样本标记预测为1或者0

使用方法：
直接使用方法 get_labeled_data_by_knn(train_data_file_path)

from label_with_KNN import get_labeled_data_by_knn
get_labeled_data_by_knn('data/small_train.csv')

"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler, MinMaxScaler


# 将20170304格式转化为2017-03-04
def changeToDate(x):
    # 20170303
    date = str(x)
    return date[0: 4]+'-' + date[4: 6] + '-' + date[-2: ]


# 参数为文件的路径，返回处理好的文件data/labeled_train_data.csv
def get_labeled_data_by_knn(train_data_file_path):
    print('knn starting....')
    print('Load data....')
    # 加载数据
    # df_train = pd.read_csv('data/small_train.csv')

    original_df_train = pd.read_csv(train_data_file_path)  # 用来保存原始的数据，只想获得未标记的数据标签
    df_train = pd.read_csv(train_data_file_path)  # 为了获取标签，用来对数据进行操作

    print("data preprocessing....")
    # 将时间拆开
    df_train['date'] = df_train['date'].apply(changeToDate)
    df_train['date'] = pd.to_datetime(df_train['date'])
    # 将date这个特征拆成三个特征，删除date
    df_train['f298'] = df_train['date'].apply(lambda x: x.year)
    df_train['f299'] = df_train['date'].apply(lambda x: x.month)
    df_train['f300'] = df_train['date'].apply(lambda x: x.day)
    # 先留着这个属性 df_train.drop('date', axis=1, inplace=True)

    # 所有的属性列
    feature_names = ['f' + str(i) for i in range(1, 301)]

    # 按照中位数补充缺失值
    df_train.fillna(df_train.median(), inplace=True)

    # 对数据进行归一化
    standard_scaler = StandardScaler()
    df_train[feature_names] = standard_scaler.fit_transform(df_train[feature_names].values)
    print("data preprocessing complete")

    k = 60
    # 训练knn
    # knn model
    knn = KNeighborsClassifier(n_neighbors=k,
                               algorithm='kd_tree',
                               leaf_size=30,
                               p=2,
                               n_jobs=-1)

    print("knn trainning....")
    print("k = ", k)
    knn.fit(X=df_train.loc[df_train['label'] != -1, feature_names].values,
            y=df_train.loc[df_train['label'] != -1, 'label'].values)

    print("knn predicting....")

    pre = knn.predict(X=df_train.loc[df_train['label'] == -1, feature_names].values)
    print(pre)
    # loc根据筛选条件  获取指定的那些列， 下面是重新赋值, 对原始的数据进行操作

    original_df_train.loc[original_df_train['label'] == -1, 'label'] = pre

    # 将标记好的数据进行保存
    original_df_train.to_csv("data/labeled_train_data.csv", encoding='utf-8', index=False)














