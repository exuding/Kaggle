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
from sklearn.model_selection import KFold
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
    pickle_file = 'trainAndTestData0920.pickle'
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

    cate = [x for x in X_train.columns if (x == 'ProductCD' or x.startswith("addr") or x.startswith("card") or
                                           x.endswith("domain") or x.startswith("Device")) and not x.endswith("count")]
    print(cate)
    params = {'application': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              'max_depth': 16,
              'learning_rate': 0.05,
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


    if not debug:
        sample_submission['isFraud'] = (y_preds11 * 0.5 + y_preds * 0.5 + y_preds2) * 0.5
        sample_submission.to_csv('simple_ensemble9999.csv')