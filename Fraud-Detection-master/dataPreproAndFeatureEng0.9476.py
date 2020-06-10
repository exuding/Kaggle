import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import time
from tqdm import tqdm_notebook
import pickle
import datetime
import gc
from itertools import cycle, islice
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit
import multiprocessing
import warnings
import datetime
from sklearn.preprocessing import LabelEncoder
from util.reduceMemory import reduce_mem_usage
from util.splitBox import ChiMerge
from util.computeWoeAndIv import valueToGroup
from util.computeWoeAndIv import computWoe
from itertools import combinations

sns.set()
warnings.simplefilter('ignore')
files = ['./input/test_identity.csv',
         './input/test_transaction.csv',
         './input/train_identity.csv',
         './input/train_transaction.csv',
         './input/sample_submission.csv']


#压缩数据
def load_data(file):
    return reduce_mem_usage(pd.read_csv(file))

# 现对train-test-identy处理设备特征
def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]
    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]
    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]
    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]
    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]
    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                 dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    gc.collect()
    return dataframe

#浏览器特征处理
def setbrowser(df):
    df.loc[df["id_31"] == "samsung browser 7.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "opera 53.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "mobile safari 10.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "google search application 49.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "firefox 60.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "edge 17.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 69.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 67.0 for android", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 63.0 for android", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 63.0 for ios", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 64.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 64.0 for android", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 64.0 for ios", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 65.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 65.0 for android", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 65.0 for ios", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 66.0", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 66.0 for android", 'lastest_browser'] = 1
    df.loc[df["id_31"] == "chrome 66.0 for ios", 'lastest_browser'] = 1
    return df

# 统计M1-M9共有多少个T
def getx(x):
    if x == 'T':
        return 1
    elif x == 'F':
        return 0

if __name__=='__main__':
    # 线程池读取多个数据集
    with multiprocessing.Pool() as pool:
        test_identity, test_transaction, train_identity, train_transaction, sample_submission = pool.map(load_data,
                                                                                                         files)

    # 各个数据集缺失情况
    print("% of train_transaction jupyter missing = ",
          (train_transaction[train_transaction.columns].isnull().sum().sum() / np.product(
              train_transaction.shape)) * 100)
    print("% of train_identity jupyter missing = ",
          (train_identity[train_identity.columns].isnull().sum().sum() / np.product(train_identity.shape)) * 100)
    print("% of test_transaction jupyter missing = ",
          (test_transaction[test_transaction.columns].isnull().sum().sum() / np.product(test_transaction.shape)) * 100)
    print("% of test_identity jupyter missing = ",
          (test_identity[test_identity.columns].isnull().sum().sum() / np.product(test_identity.shape)) * 100)

    #对identity数据集 设备特征 进行处理
    train_identity = id_split(train_identity)
    test_identity = id_split(test_identity)
    # 根据设备名设置是否是可信设备 0909 add by taoxudong
    for df in [train_identity, test_identity]:
        df["had_id"] = [0 if status == 'Others' else 1 for status in df["device_name"].apply(str)]

    # 合并数据
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
    del test_identity, test_transaction, train_identity, train_transaction
    #垃圾回收机制
    gc.collect()

    # 预处理
    # 第一步删除两个数据集中超过0.9的空值特征
    train_missing_values = train.isnull().sum().sort_values(ascending=False) / len(train)
    test_missing_values = test.isnull().sum().sort_values(ascending=False) / len(test)
    #缺失率>0.9删除
    train_missing_values = [str(x) for x in train_missing_values[train_missing_values > 0.90].keys()]
    test_missing_values = [str(x) for x in test_missing_values[test_missing_values > 0.90].keys()]

    dropped_columns = train_missing_values + test_missing_values
    # # 第二步删除特征分布不均匀的情况二  即特征值超过0.9都是同一个值     但是isFraud也属于这种情况要去除掉
    dropped_columns = dropped_columns + [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    dropped_columns = dropped_columns + [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    dropped_columns.remove('isFraud')
    # 删除不符合的特征
    train.drop(dropped_columns, axis=1, inplace=True)
    test.drop(dropped_columns, axis=1, inplace=True)
    print('删除的列的个数:',len(dropped_columns))
    #-----------------------------可视化-----------------------------------------
    # 查看分布
    # train["DeviceType"].value_counts(dropna=False).plot.bar()
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # sns.barplot(train["DeviceInfo"].value_counts(dropna=False)[:15],
    #             train["DeviceInfo"].value_counts(dropna=False).keys()[:15])
    # plt.show()

    # train["DeviceType"].value_counts(dropna=False).plot.bar()
    # plt.show()

    # 使用最新浏览器的用户进行的交易中有10.7%是欺诈行为。
    # plt.figure(figsize=(6, 6))
    # plt.pie([np.sum(train[(train['lastest_browser'] == True)].isFraud.values),
    #          len(train[(train['lastest_browser'] == True)].isFraud.values) -
    #          np.sum(train[(train['lastest_browser'] == True)].isFraud.values)],
    #         labels=['isFraud', 'notFraud'], autopct='%1.1f%%', colors=['y', 'g'])
    # plt.show()
    #-----------------------------可视化--------------------------------------------
    ########################### Device info
    for df in [train, test]:
        ########################### Device info
        df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
        df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
        ########################### Device info 2
        df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
        df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
        ########################### Browser
        df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
        df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    # 转化成时间格式
    startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
    for df in [train, test]:
        # Temporary
        df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        df['DT_M'] = (df['TransactionDT'].dt.year - 2017) * 12 + df['TransactionDT'].dt.month
        df['DT_W'] = (df['TransactionDT'].dt.year - 2017) * 52 + df['TransactionDT'].dt.weekofyear
        df['DT_D'] = (df['TransactionDT'].dt.year - 2017) * 365 + df['TransactionDT'].dt.dayofyear
        df['DT_hour'] = df['TransactionDT'].dt.hour
        df['DT_day_week'] = df['TransactionDT'].dt.dayofweek
        df['DT_day'] = df['TransactionDT'].dt.day

    # Let's add some kind of client uID based on cardID ad addr columns
    train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
    test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)
    train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train['card5'].astype(str)
    test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test['card5'].astype(str)
    train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['addr2'].astype(str)
    test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test['addr2'].astype(str)

    ########################### Freq encoding
    i_cols = ['card1', 'card2', 'card3', 'card5',
              'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8',
              'addr1', 'addr2',
              'dist1',
              'P_emaildomain', 'R_emaildomain',
              'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
              'id_30', 'id_30_device', 'id_30_version',
              'id_31_device',
              'id_33',
              'uid', 'uid2', 'uid3',
              ]
    for col in i_cols:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        train[col + '_fq_enc'] = train[col].map(fq_encode)
        test[col + '_fq_enc'] = test[col].map(fq_encode)

    for col in ['DT_M', 'DT_W', 'DT_D']:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        train[col + '_total'] = train[col].map(fq_encode)
        test[col + '_total'] = test[col].map(fq_encode)

    periods = ['DT_M', 'DT_W', 'DT_D']
    i_cols = ['uid']
    for period in periods:
        for col in i_cols:
            new_column = col + '_' + period

            temp_df = pd.concat([train[[col, period]], test[[col, period]]])
            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
            fq_encode = temp_df[new_column].value_counts().to_dict()

            train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)
            test[new_column] = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

            train[new_column] /= train[period + '_total']
            test[new_column] /= test[period + '_total']

    # 邮箱处理
    train['is_proton_mail'] = (train['P_emaildomain'] == 'protonmail.com') | (train['R_emaildomain'] == 'protonmail.com')
    test['is_proton_mail'] = (test['P_emaildomain'] == 'protonmail.com') | (test['R_emaildomain'] == 'protonmail.com')
    train['is_mail'] = (train['P_emaildomain'] == 'mail.com') | (train['R_emaildomain'] == 'mail.com')
    test['is_mail'] = (test['P_emaildomain'] == 'mail.com') | (test['R_emaildomain'] == 'mail.com')

    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
              'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
              'hotmail.de': 'microsoft',
              'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
              'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
              'protonmail.com': 'other',
              'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
              'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
              'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
              'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
              'ptd.net': 'other',
              'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        # 公司 google
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)
        # 组织 com
        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    print('email_process_done')

    #浏览器特征处理--------------
    a = np.zeros(train.shape[0])
    train["lastest_browser"] = a
    a = np.zeros(test.shape[0])
    test["lastest_browser"] = a
    train = setbrowser(train)
    test = setbrowser(test)
    # #############-------------

    # feature interaction
    for card in ['card1_card2', 'card1_card3', 'card1_addr2', 'card2_addr1', 'card2_addr2', 'addr1_addr2',
                 'ProductCD_card1',
                 'ProductCD_card2', 'ProductCD_card3', 'ProductCD_card4', 'ProductCD_card5', 'ProductCD_addr1',
                 'ProductCD_addr2']:
        card1 = card.split('_')[0]
        card2 = card.split('_')[1]
        train[card] = train[card1].astype(str) + '_' + train[card2].astype(str)
        test[card] = test[card1].astype(str) + '_' + test[card2].astype(str)

        train[card + '_amt_mean'] = train[card].map(
            (pd.concat([train[[card, 'TransactionAmt']], test[[card, 'TransactionAmt']]], ignore_index=True)).groupby(
                [card])['TransactionAmt'].mean())
        test[card + '_amt_mean'] = test[card].map(
            (pd.concat([train[[card, 'TransactionAmt']], test[[card, 'TransactionAmt']]], ignore_index=True)).groupby(
                [card])['TransactionAmt'].mean())

        train[card + '_amt_std'] = train[card].map(
            (pd.concat([train[[card, 'TransactionAmt']], test[[card, 'TransactionAmt']]], ignore_index=True)).groupby(
                [card])['TransactionAmt'].std())
        test[card + '_amt_std'] = test[card].map(
            (pd.concat([train[[card, 'TransactionAmt']], test[[card, 'TransactionAmt']]], ignore_index=True)).groupby(
                [card])['TransactionAmt'].std())

    ########################### ProductCD and M4 Target mean
    for col in ['ProductCD', 'M4']:  # 除M4是分类数据 其他M1-9都是二分类数据
        temp_dict = train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
        temp_dict.index = temp_dict[col].values
        temp_dict = temp_dict[col + '_target_mean'].to_dict()

        train[col + '_target_mean'] = train[col].map(temp_dict)
        test[col + '_target_mean'] = test[col].map(temp_dict)

    # Some arbitrary features interaction 0903新加特征
    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2',
                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1',
                    'id_01__card1', 'id_01__card3','card3__card5','ProductCD__TransactionAmt']:
        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    # Create Features based on anonymised prefix groups
    for df in [train, test]:
        prefix = ['C', 'D', 'Device', 'M', 'V', 'addr', 'card', 'dist', 'id']
        for i, p in enumerate(prefix):
            column_set = [x for x in df.columns.tolist() if x.startswith(prefix[i])]

            # Take NA count
            df[p + "group_nan_sum"] = df[column_set].isnull().sum(axis=1) / df[column_set].shape[1]

            # Take SUM/Mean if numeric
            numeric_cols = [x for x in column_set if df[x].dtype != object]
            if numeric_cols:
                df[p + "group_sum"] = df[column_set].sum(axis=1)
                df[p + "group_mean"] = df[column_set].mean(axis=1)
                # Zero Count
                df[p + "group_0_count"] = (df[column_set] == 0).astype(int).sum(axis=1) / (
                        df[column_set].shape[1] - df[p + "group_nan_sum"])

    # --------------------------------特征交叉----------------------------
    # 其他衍生变量 mean 和 std
    for df in [train, test]:
        for item in ['card1', 'card4']:
            df['id_02_to_mean_' + item] = df['id_02'] / df.groupby([item])['id_02'].transform('mean')
            df['id_02_to_std_' + item] = df['id_02'] / df.groupby([item])['id_02'].transform('std')

    for df in [train, test]:
        for item in ['card1', 'card4','addr1']:
            df['D15_to_mean_' + item] = df['D15'] / df.groupby([item])['D15'].transform('mean')
            df['D15_to_std_' + item] = df['D15'] / df.groupby([item])['D15'].transform('std')


    # Check if the Transaction Amount is common or not (we can use freq encoding here)
    # In our dialog with a model we are telling to trust or not to these values
    train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
    test['TransactionAmt_check'] = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)
    # New feature - decimal part of the transaction amount 小数点代表不同区域
    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
    # 其他衍生变量 mean 和 std #todo
    for df in [train,test]:
        for item in ['card1','card2','card3','card4','card5','D1','D2','C1','D4']:
            df['TransactionAmt_to_mean_'+item] = df['TransactionAmt'] / df.groupby([item])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_std_'+item] = df['TransactionAmt'] / df.groupby([item])['TransactionAmt'].transform('std')

    # 二次衍生变量
    for df in [train, test]:
        for item in ['uid', 'uid2', 'uid3']:
            df['TransactionAmt_to_mean_' + item] = df['TransactionAmt'] / df.groupby([item])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_std_' + item] = df['TransactionAmt'] / df.groupby([item])['TransactionAmt'].transform('std')

    # New feature - log of transaction amount. ()
    train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
    test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

    ########################### M columns (except M4)
    # All these columns are binary encoded 1/0
    # We can have some features from it
    i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    # 统计M1-M9共有多少个T
    for df in [train, test]:
        aa = df[i_cols].applymap(getx)
        df['M_sum'] = aa.sum(axis=1).astype(np.int8)
        df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

    i_cols = ['C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for df in [train, test]:
        df['C_sum'] = df[i_cols].sum(axis=1).astype(np.int8)

    # 对连续性变量D1-15做分箱处理 0910 by taoxudong

    ###########函数计算
    # 填充空值
    # dlist = ['D2','D6']
    # for item in dlist:
    #     train[item] = train[item].fillna(-999)
    #     test[item] = test[item].fillna(-999)
    #     # 卡方分箱法对变量进行分箱 得到分箱点
    #     cutoffs = ChiMerge(train, item, 'isFraud', max_interval=6)
    #     # cutoffs 切分点集合
    #     train[item + '_group'] = train[item].apply(valueToGroup, args=(cutoffs,))
    #     test[item + '_group'] = test[item].apply(valueToGroup, args=(cutoffs,))
    #     # woe和iv后面用到再释放出来
    #     woe_map = computWoe(train,item+'_group','isFraud')
    #     print('woe_map{}'.format(item),woe_map)
        # iv = computIV(train,'D1_group','isFraud')
        # train['D1_woe'] = train['D1_group'].map(woe_map)


    # 卡方分箱法对变量进行分箱 得到分箱点
    cutoffs = ChiMerge(train, 'TransactionAmt', 'isFraud', max_interval=6)
    # cutoffs 切分点集合
    train['TransactionAmt_group'] = train.TransactionAmt.apply(valueToGroup, args=(cutoffs,))
    test['TransactionAmt_group'] = test.TransactionAmt.apply(valueToGroup, args=(cutoffs,))


    for df in [train,test]:
        # D9 column
        df['D9'] = np.where(df['D9'].isna(), 0, 1)

    # 变量用count做衍生  计数编码
    # card的 #变量用count做衍生  计数编码
    for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_34', 'id_36']:
        train[feature + '_count_full'] = train[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[feature + '_count_full'] = test[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    # D的 #变量用count做衍生  计数编码 #todo
    # for feature in  ['D1','D2','D4','D11','D13']:
    #     train[feature + '_count_full'] = train[feature].map(
    #         pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    #     test[feature + '_count_full'] = test[feature].map(
    #         pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    print('count计数完成')

    # Encoding - count encoding separately for train and test
    for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
    ########################### Reset values for "noise" card1
    for col in ['card1']:
        valid_card = pd.concat([train[[col]], test[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
        test[col] = np.where(test[col].isin(train[col]), test[col], np.nan)

        train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)
        test[col] = np.where(test[col].isin(valid_card), test[col], np.nan)

    # add by taoxudong 特征重要性中前20的连续特征进行交叉
    # serialFeatureTop = ['C13', 'C1', 'D4', 'TransactionAmt', 'C2', 'C14', 'C6', 'C11','C13']
    # #TransactionAmt*C2  C1*TransactionAmt	TransactionAmt/C6  TransactionAmt*C11  C13/TransactionAmt  C1/C2  C1/TransactionAmt
    # #C13/C6	C13/C2  C13/C11  C2/C11  C13/C1  C13*TransactionAmt C11/C11
    # #缺失值的填充
    # for df in [train, test]:
    #     for column in serialFeatureTop:
    #         mean_val = df[column].mean()
    #         df[column].fillna(mean_val, inplace=True)
    # #1排列组合交叉生成
    # f2_columns = ['C13', 'C1', 'D4', 'C2', 'C6', 'C11','C13']
    # for df in [train, test]:
    #     for f2 in f2_columns:
    #         df['TransactionAmt' + '/' + f2] = df['TransactionAmt'] / df[f2]
    #         df['TransactionAmt' + '*' + f2] = df['TransactionAmt'] * df[f2]
    # #2排列组合交叉生成
    # f3_columns = ['C1', 'C6', 'C11','C2','D2','D10']
    # for df in [train, test]:
    #     for f3 in f3_columns:
    #         df['C13' + '/' + f3] = df['C13'] / df[f3]
    #         df['C13' + '*' + f3] = df['C13'] * df[f3]
    # #3排列组合交叉生成
    # for df in [train, test]:
    #     df['C1' + '/' + 'C2'] = df['C1'] / df['C2']
    #     df['C1' + '/' + 'C6'] = df['C1'] / df['C6']
    #     df['C14' + '/' + 'C11'] = df['C14'] / df['C11']
    #     df['C2' + '/' + 'C6'] = df['C2'] / df['C6']
    # print('连续特征融合完成')

    # 填充掉空值
    numerical_columns = list(test.select_dtypes(exclude=['object']).columns)
    train[numerical_columns] = train[numerical_columns].fillna(train[numerical_columns].median())
    test[numerical_columns] = test[numerical_columns].fillna(train[numerical_columns].median())
    print("filling numerical columns null values done")

    # Now, we find out categorical columns
    categorical_columns = list(filter(lambda x: x not in numerical_columns, list(test.columns)))
    print(categorical_columns[:5])

    # then, fill missing values in categorical columns
    train[categorical_columns] = train[categorical_columns].fillna(train[categorical_columns].mode())
    test[categorical_columns] = test[categorical_columns].fillna(train[categorical_columns].mode())
    print("filling categorica columns null values done")


    # Label Encoding 转化X_train[f].dtype=='object'的列
    # 标准化标签，将标签值统一转换成range(标签值个数-1)范围
    for col in tqdm_notebook(categorical_columns):
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    print('Label Encoding 完成')
    # Because of using cross validation, we use this step later when we want to find best model. Now, we have to remove isFraud column from daraframe
    labels = train["isFraud"]
    train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1, inplace=True)
    test.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)

    print(train.shape,test.shape)

    X_train, y_train = train, labels
    # del train, labels
    # gc.collect()

    # 保存数据方便调用
    pickle_file = 'trainAndTestData.pickle'
    if not os.path.isfile(pickle_file):  # 判断是否存在此文件，若无则存储
        print('Saving jupyter to pickle file...')
        try:
            with open('trainAndTestData.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': X_train,
                        'train_labels': y_train,
                        'test_dataset': test,
                        'valid_dataset':sample_submission
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save jupyter to', pickle_file, ':', e)
            raise

