#-*- coding:utf-8 -*-
'''
@project: Exuding
@author: taoxudong
@time: 2019-11-29 11:42:29
'''
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from urllib.parse import urlparse
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from category_encoders.one_hot import OneHotEncoder
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import ReLU
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import MSELoss
from torch.optim import Adam

import random

from scipy.stats import spearmanr

from sklearn.model_selection import KFold
import math

class PyTorch:

    def __init__(self, in_features, out_features, n_epochs, patience):
        self.in_features = in_features
        self.out_features = out_features
        self.n_epochs = n_epochs
        self.patience = patience

    def init_model(self):

        # define a models
        self.model = Sequential(
            weight_norm(Linear(self.in_features, 128)),
            ReLU(),
            weight_norm(Linear(128, 128)),
            ReLU(),
            weight_norm(Linear(128, self.out_features)))

        # initialize models
        # he initialization
        for t in self.model:
            if isinstance(t, Linear):
                #参数初始化
                nn.init.kaiming_normal_(t.weight_v)#在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0
                nn.init.kaiming_normal_(t.weight_g)
                #在非线性激活函数之前，我们想让输出值有比较好的分布（例如高斯分布），以便于计算梯度和更新参数。
                #Batch Normalization 将输出值强行做一次 Gaussian Normalization 和线性变换：
                nn.init.constant_(t.bias, 0)#Batchnorm Initialization


        # Xavier Initialization
        '''
        for m in self.models:
            if isinstance(m,Linear):
                nn.init.xavier_uniform(m.weight)
        '''
        # Orthogonal Initialization
        '''
        for m in self.models:
            if isinstance(m,Linear):
                nn.init.orthogonal(m.weight)
        '''




        # define loss function
        self.loss_func = MSELoss()

        # define optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def fit(self, x_train, y_train, x_valid, y_valid):

        self.init_model()

        x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32)
        y_valid_tensor = torch.as_tensor(y_valid, dtype=torch.float32)

        min_loss = np.inf
        counter = 0

        for epoch in range(self.n_epochs):

            self.model.train()
            y_pred = self.model(x_train_tensor)
            loss = self.loss_func(y_pred, y_train_tensor)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss = loss.item()
            # print('Epoch %5d / %5d. Loss = %.5f' % (epoch + 1, self.n_epochs, epoch_loss))

            # calculate loss for validation set
            self.model.eval()
            with torch.no_grad():
                valid_loss = self.loss_func(self.model(x_valid_tensor), y_valid_tensor).item()

            # print('Epoch %5d / %5d. Validation loss = %.5f' % (epoch + 1, self.n_epochs, valid_loss))

            # early stopping
            if valid_loss < min_loss:
                min_loss = valid_loss
                counter = 0
            else:
                counter += 1
                # print('Early stopping: %i / %i' % (counter, self.patience))
                if counter >= self.patience:
                    # print('Early stopping at epoch', epoch + 1)
                    break

    def predict(self, x):
        x_tenson = torch.as_tensor(x, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(x_tenson).numpy()

def mean_spearmanr_correlation_score(y_true, y_pred):
    return np.mean([spearmanr(y_pred[:, idx], y_true[:, idx]).correlation for idx in range(len(target_columns))])

def make_user_map(user_pages):
    user_map = {}
    for p in user_pages:
        # get groups from URL (https://)(photo.stackexchange)(.com/users/1024)
        a = re.search('(https:\/\/)(.*)(.com|.net)', p)
        if a:
            # get second group (photo).(stackexchange) or use whole site
            b = re.search('(.*\.)(.*)', a.group(2))
            if b:
                s = b.group(2)
            else:
                s = a.group(2)
            # get user id from (https://photo.stackexchange.com/users/)(1024)
            c = re.search('(.*\/)(\d*)', p)
            if c:
                u = c.group(2)
            else:
                u = 'unknown'
            user_map[p] = s + '_' + u
    return user_map

if __name__=='__main__':
    warnings.simplefilter('ignore')

    train = pd.read_csv("./input/train.csv", index_col='qa_id')
    test = pd.read_csv("./input/test.csv", index_col='qa_id')
    '''
    Let's make user id from site and user id assuming user has the same id on all stackexchange sites.
    For example, for URL https://photo.stackexchange.com/users/1024 our id will be stackexchange_1024
    '''
    train_question_user_pages = train['question_user_page'].unique()
    train_question_user_map = make_user_map(train_question_user_pages)
    train['question_user'] = train['question_user_page'].apply(lambda x: train_question_user_map[x])

    train_answer_user_pages = train['answer_user_page'].unique()
    train_answer_user_map = make_user_map(train_answer_user_pages)
    train['answer_user'] = train['answer_user_page'].apply(lambda x: train_answer_user_map[x])

    test_question_user_pages = test['question_user_page'].unique()
    test_question_user_map = make_user_map(test_question_user_pages)
    test['question_user'] = test['question_user_page'].apply(lambda x: test_question_user_map[x])

    test_answer_user_pages = test['answer_user_page'].unique()
    test_answer_user_map = make_user_map(test_answer_user_pages)
    test['answer_user'] = test['answer_user_page'].apply(lambda x: test_answer_user_map[x])

    target_columns = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'
    ]
    y_train = train[target_columns].copy()
    x_train = train.drop(target_columns, axis=1)
    del train
    x_test = test.copy()
    del test

    ##Pipeline可以直接调用fit和predict方法来对pipeline中的所有算法模型进行训练和预测/结合GridSearch来对参数进行选择：
    #TF-IDF + SVD for text features
    text_encoder = Pipeline([
        ('Text-TF-IDF', TfidfVectorizer(ngram_range=(1, 3))),
        ('Text-SVD', TruncatedSVD(n_components=100))], verbose=True)

    #Encode 'url'
    # gives part of string (URL) before '.'
    before_dot = re.compile('^[^.]*')
    def transform_url(x):
        return x.apply(lambda v: re.findall(before_dot, urlparse(v).netloc)[0])

    url_encoder = Pipeline([
        ('URL-transformer', FunctionTransformer(transform_url, validate=False)),
        ('URL-OHE', OneHotEncoder(drop_invariant=True))], verbose=True)
    #Encode 'category'
    ohe = OneHotEncoder(cols='category', drop_invariant=True)
    #Transform
    preprocessor = ColumnTransformer([
        ('Q-T', text_encoder, 'question_title'),
        ('Q-B', text_encoder, 'question_body'),
        ('A', text_encoder, 'answer'),
        ('URL', url_encoder, 'url'),
        ('Categoty', ohe, 'category')], verbose=True)

    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    print(x_train.shape)
    y_train = y_train.values
    ###########################train####################
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    pytorch_params = {
        'in_features': x_train.shape[1],
        'out_features': y_train.shape[1],
        'n_epochs': 2500,
        'patience': 5
    }

    n_splits = 10

    trained_estimators = []
    scores = []

    cv = KFold(n_splits=n_splits, random_state=42)
    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train[train_idx]
        y_train_train = y_train[train_idx]
        x_train_valid = x_train[valid_idx]
        y_train_valid = y_train[valid_idx]

        estimator = PyTorch(**pytorch_params)
        estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)

        oof_part = estimator.predict(x_train_valid)
        score = mean_spearmanr_correlation_score(y_train_valid, oof_part)
        print('Score:', score)

        if not math.isnan(score):
            trained_estimators.append(estimator)
            scores.append(score)

    print('Mean score:', np.mean(scores))


    y_pred = []
    for estimator in trained_estimators:
        y_pred.append(estimator.predict(x_test))
    #Blend by ranking
    from scipy.stats import rankdata
    def blend_by_ranking(data, weights):
        out = np.zeros(data.shape[0])
        for idx, column in enumerate(data.columns):
            out += weights[idx] * rankdata(data[column].values)
        out /= np.max(out)
        return out

    submission = pd.read_csv("./input/sample_submission.csv", index_col='qa_id')

    out = pd.DataFrame(index=submission.index)
    for column_idx, column in enumerate(target_columns):

        # collect all predictions for one column
        column_data = pd.DataFrame(index=submission.index)
        for prediction_idx, prediction in enumerate(y_pred):
            column_data[str(prediction_idx)] = prediction[:, column_idx]

        out[column] = blend_by_ranking(column_data, np.ones(column_data.shape[1]))

    out.to_csv("submission.csv")