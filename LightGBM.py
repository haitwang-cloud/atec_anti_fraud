# -*- coding: utf-8 -*-
"""
@date: Created on Sun May 13 22:37:21 2018
@author: zhaoguangjun
@desc: 训练lightgbm分类器
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_curve 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

from operator import itemgetter
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import time
from time import strftime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from time import strftime
import gc
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def score(y_true, y_score):
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    score = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.01)[0][0]]

    return score


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

def evaluate(y_true, y_pred, y_prob):
    """ 估计结果: precision, recall, f1, auc, mayi_score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mayi = score(y_true, y_prob)
    
    return [p,r,f1,auc,mayi]    
    

def kfold_model_train(clf_fit_params, clf, X, y, n_splits=5):
    """ 进行K-fold模型训练
    """
    models, i = [], 0
    eval_train = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    eval_test  = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X)):
        X_, X_test = X[train_index], X[test_index]
        y_, y_test = y[train_index], y[test_index]        
        
        # Divide into train and validation set for early-stop
        X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=0.15, random_state=42)
        
        del X_, y_
        gc.collect()
        
        # Model Training
        clf.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc',
                verbose=False, **clf_fit_params)
        
        ## Model Testing
        # On training set
        y_prob_train = clf.predict_proba(X_train)[:,1]
        y_pred_train = clf.predict(X_train)
        eval_train.iloc[i,:] = evaluate(y_train, y_pred_train, y_prob_train)
        
        # On testing set
        y_prob_test = clf.predict_proba(X_test)[:,1]
        y_pred_test = clf.predict(X_test)
        eval_test.iloc[i,:] = evaluate(y_test, y_pred_test, y_prob_test)
        
        # Saving model
        models.append(clf)
        i += 1
        
    return models, eval_train, eval_test
        
#%%    
if __name__ == '__main__':
    # 加载数据
    with timer('Load Data'):
        train = pd.read_csv('./dataset/atec_anti_fraud_train.csv',encoding='utf-8',low_memory=False,parse_dates=['date'])
        train=train[train['label']!=-1]
    with timer('Model Train'):
        feature_name = ['f'+str(i) for i in range(1,298)] # 所有变量的名称
        nunique = train[feature_name].nunique()  # 每个特征分量unique值的数量
        categorical_feature = list(nunique[nunique <= 10].index.values) # 所有类别变量的名称
        
        # 训练样本以及类别标签
        X, y = train[feature_name].values, train['label'].values
        
        # 构造分类器
        lgb_params = {'boosting_type': 'gbdt',
                      'num_leaves': 31,
                      'max_depth': 50,
                      'learning_rate': 0.10,
                      'n_estimators': 100000,
                      'reg_alpha': 0.1,
                      'seed': 42,
                      'nthread': -1}
        
        clf = lgb.LGBMClassifier(**lgb_params)
        
        # 分类器训练
        clf_fit_params = {'early_stopping_rounds': 5, 'feature_name': feature_name,
                          'categorical_feature': categorical_feature}
        
        models, eval_train, eval_test = kfold_model_train(clf_fit_params, clf, X, y, n_splits=5)

    with timer('Mode test'):
        test = pd.read_csv('./dataset/atec_anti_fraud_test_a.csv',encoding='utf-8',low_memory=False,parse_dates=['date'])
        X_test = preprocess(test[feature_name].copy()) 
                
        test_prob_final = np.zeros((len(X_test),))
        for model in models:
            test_prob = model.predict_proba(X_test)[:,1]
            test_prob_final += (test_prob*0.2)

    with timer('Write Result'):
        result = pd.DataFrame()
        result['id'] = test['id']
        result['score'] = test_prob_final
        result.to_csv('./dataset/submission_180514_v4.csv', index=False)

        result['pred'] = (result['score'] > 0.5).astype('int')





