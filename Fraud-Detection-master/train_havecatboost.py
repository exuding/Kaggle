#-*- coding:utf-8 -*-
'''
@project: Fraud-Detection
@author: taoxudong
@time: 2019-09-18 15:09:47 
'''

import pickle
import gc
import xgboost as xgb
import numpy as np
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, Pool
import os
import random
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

if __name__=='__main__':
    # 读取数据
    pickle_file = 'trainAndTestData0.9476.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  # 反序列化，与pickle.dump相反
        X_train = pickle_data['train_dataset']
        y_train = pickle_data['train_labels']
        test = pickle_data['test_dataset']
        sample_submission = pickle_data['valid_dataset']

        del pickle_data  # 释放内存

    debug = False
    if debug:
        split_pos = X_train.shape[0] * 4 // 5
        y_test = y_train.iloc[split_pos:]
        y_train = y_train.iloc[:split_pos]
        test = X_train.iloc[split_pos:, :]
        X_train = X_train.iloc[:split_pos, :]


    #-------------------------------第一次-------------------------------------------

    folds = 3
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_preds = np.zeros(test.shape[0])
    i = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):
        i += 1
        clf = xgb.XGBClassifier(
            n_estimators=700,
            max_depth=9,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method='gpu_hist'
        )

        X_tr = X_train.iloc[tr_idx, :]
        y_tr = y_train.iloc[tr_idx]
        clf.fit(X_tr, y_tr)
        del X_tr
        y_preds += clf.predict_proba(test)[:, 1] / folds
        if debug:
            print("debug:", roc_auc_score(y_test, clf.predict_proba(test)[:, 1] / folds))
        del clf

    if debug:
        print("debug:", roc_auc_score(y_test, y_preds))

    gc.collect()


    features = [x for x in X_train.columns]

    folds = 3
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_preds11 = np.zeros(test.shape[0])
    i = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):
        i += 1
        clf = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=9,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method='gpu_hist'
        )

        X_tr = X_train[features].iloc[tr_idx, :]
        y_tr = y_train.iloc[tr_idx]
        clf.fit(X_tr, y_tr)
        del X_tr
        y_preds11 += clf.predict_proba(test[features])[:, 1] / folds
        if debug:
            print("debug:", roc_auc_score(y_test, clf.predict_proba(test[features])[:, 1] / folds))
        del clf

    gc.collect()
    if debug:
        print("debug:", roc_auc_score(y_test, y_preds11))
        print("debug:", roc_auc_score(y_test, y_preds11 * 0.5 + y_preds * 0.5))
    #-------------------------------第二次-------------------------------------------
    cate = [x for x in X_train.columns if (x == 'ProductCD' or x.startswith("addr") or x.startswith("card") or
                                           x.endswith("domain") or x.startswith("Device")) and not x.endswith("count")]
    print(cate)
    params = {'application': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              'max_depth': 16,
              'learning_rate': 0.01,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.9,
              'verbosity': -1,
              'lambda_l1': 0.1,
              'lambda_l2': 0.01,
              'num_leaves': 500,
              'min_child_weight': 3,
              'data_random_seed': 17,
              'nthreads': 4}

    early_stop = 500
    verbose_eval = 30
    num_rounds = 600

    #
    folds = 3
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_preds2 = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    i = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):

        X_tr = X_train.iloc[tr_idx, :]
        y_tr = y_train.iloc[tr_idx]
        d_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cate)
        watchlist = []
        if debug:
            d_test = lgb.Dataset(test, label=y_test, categorical_feature=cate)
            watchlist.append(d_test)

        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval)

        y_preds2 += model.predict(test) / folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = X_tr.columns
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if debug:
            print("debug:", roc_auc_score(y_test, model.predict(test) / folds))
        i += 1
        del X_tr, d_train

    if debug:
        print("debug:", roc_auc_score(y_test, y_preds2))
        print("debug:", roc_auc_score(y_test, (y_preds + y_preds2) * 0.5))

    if debug:
        print("debug:", roc_auc_score(y_test, y_preds))
        print("debug:", roc_auc_score(y_test, y_preds2))
        print("debug:", roc_auc_score(y_test, (y_preds + y_preds2) * 0.5))

    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:100].index)
    print(cols)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    #---------------------------------------第三次-------------------------------
    # 标准化标签，将标签值统一转换成range(标签值个数-1)范围  将字符串类型转化为数字categorical_feature后面才能用
    categorical_feature = ['ProductCD',
                           'P_emaildomain', 'R_emaildomain',
                           'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                           'addr1', 'addr2',
                           'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    for col in tqdm_notebook(categorical_feature):
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    print('Label Encoding 完成')

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

    # 3折交叉验证
    folds = TimeSeriesSplit(n_splits=5)
    for fold, (trn_idx, test_idx) in enumerate(folds.split(X_train, y_train)):
        print('Training on fold {}'.format(fold + 1))
        # 构建成lgb特征的数据集格式 因为是CV所以用索引划分训练集和验证集
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx],
                               categorical_feature=categorical_feature)
        val_data = lgb.Dataset(X_train.iloc[test_idx], label=y_train.iloc[test_idx],
                               categorical_feature=categorical_feature)
        # 参数写成字典形式后 训练
        clf = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=500)
        # 获取最重要的特征
    print('-' * 30)
    print('Training has finished.')
    print('-' * 30)
    # 重新挑出最优模型来预测
    best_iter = clf.best_iteration
    clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
    clf.fit(X_train, y_train)
    # 预测
    y_preds3 = clf.predict_proba(test)[:, 1]

    #--------------------------第五个------------------------------

    import catboost as cb
    from catboost import CatBoostClassifier, Pool

    features = [x for x in X_train.columns]

    cate = [x for x in X_train.columns if (x == 'ProductCD' or x in ['card1', 'card2'] or x.startswith("addr") or
                                           x.endswith("domain") or x.startswith("Device")) and not x.endswith(
        "count") and not x == "id_11"]

    # cate = []
    print(cate)
    verbose_eval = 30
    num_rounds = 800

    folds = 3
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    y_preds4 = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    i = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):

        X_tr = X_train[features].iloc[tr_idx, :].fillna(-1)
        y_tr = y_train.iloc[tr_idx]

        model = cb.CatBoostClassifier(iterations=num_rounds, depth=14, learning_rate=0.04, loss_function='Logloss',
                                      eval_metric='Logloss'
                                      , task_type="CPU")
        if debug:
            model.fit(X_tr, y_tr, cat_features=cate, verbose_eval=30)
        else:
            model.fit(X_tr, y_tr, cat_features=cate, verbose_eval=30)

        del X_tr
        y_preds4 += model.predict_proba(test[features].fillna(-1))[:, 1] / folds

    if not debug:
        sample_submission['isFraud'] = (y_preds11 * 0.5 + y_preds * 0.5 + y_preds2*0.5 + y_preds3*0.5 +y_preds4) * 0.333
        sample_submission.to_csv('mojinxiaowei.csv')