#-*- coding:utf-8 -*-
'''
@project: IEEE-CIS-Fraud-Detection-LGB-master
@author: taoxudong
@time: 2019-09-12 17:33:58 
'''
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import lightgbm as lgb
from sklearn.model_selection import KFold, TimeSeriesSplit
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder




if __name__=='__main__':
    # 读取数据
    pickle_file = 'trainAndTestData0925.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  # 反序列化，与pickle.dump相反
        X_train = pickle_data['train_dataset']
        y_train = pickle_data['train_labels']
        test = pickle_data['test_dataset']
        sample_submission = pickle_data['valid_dataset']

        del pickle_data  # 释放内存

    params = {
        'learning_rate': 0.01,  # 0.05
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'feature_fraction': 0.9,  # 降低过拟合
        'bagging_fraction': 0.9,  # 降低过拟合  相当于subsample样本采样
        'bagging_freq': 2,  # 和bagging_fraction必须同时设置， 每两轮迭代进行一次bagging
        'num_leaves': 389,  # 150 小于2的max_depth次方
        'min_data_in_leaf': 106,
        'verbose': -1,
        'max_depth': 9,
        'lambda_l2': 0.5, 'lambda_l1': 0,
        'nthread': -1,
        'seed': 89,
        # 'n_estimators':200, ####加这个估计器数目 准确度就下来了
        # 'tree_method': 'gpu_hist',
        # 'device': 'gpu', ##如果安装的事gpu版本的lightgbm,可以加快运算
        # 'gpu_platform_id': 1,# Intel，Nvidia，AMD
        # 'gpu_device_id': 0
    }

    # 五折交叉验证
    folds = TimeSeriesSplit(n_splits=5)

    aucs = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X_train.columns
    categorical_feature = ['ProductCD',
     'P_emaildomain','R_emaildomain',
     'card1','card2','card3','card4','card5','card6',
     'addr1','addr2',
     'M1','M2','M3','M4','M5','M6','M7','M8','M9']
    # 标准化标签，将标签值统一转换成range(标签值个数-1)范围  将字符串类型转化为数字categorical_feature后面才能用
    #performing Label Encoding below, you must encode train and test together
    for col in tqdm_notebook(categorical_feature):
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    print('Label Encoding 完成')

    training_start_time = time()
    for fold, (trn_idx, test_idx) in enumerate(folds.split(X_train, y_train)):
        start_time = time()
        print('Training on fold {}'.format(fold + 1))
        #构建成lgb特征的数据集格式 因为是CV所以用索引划分训练集和验证集
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx],categorical_feature=categorical_feature)
        val_data = lgb.Dataset(X_train.iloc[test_idx], label=y_train.iloc[test_idx],categorical_feature=categorical_feature)
        #参数写成字典形式后 训练
        clf = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=500)
        #获取最重要的特征
        feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
        aucs.append(clf.best_score['valid_1']['auc'])
        print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    print('-' * 30)
    print('Training has finished.')
    print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
    print('Mean AUC:', np.mean(aucs))
    print('-' * 30)

    # 获取重要特征------------------------------------求平均----------------------------------------------------------------
    feature_importances['average'] = feature_importances[
        ['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
    feature_importances.to_csv('./output/feature_importances.csv')

    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
    plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
    plt.show()
    #重新挑出最优模型来预测
    best_iter = clf.best_iteration
    clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
    clf.fit(X_train, y_train)
    #预测
    sample_submission['isFraud'] = clf.predict_proba(test)[:, 1]

    sample_submission.to_csv('./output/aaa0920.csv', index=False)
