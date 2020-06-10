# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import datetime

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
gc.enable()

import seaborn as sns
import matplotlib.pyplot as plt
from time import time, strftime, localtime

def show_cur_time():
    print(strftime('%Y-%m-%d %H:%M:%S',localtime(time())))
    
# some helps to discover feature interaction
def cat_dist_target_show(train, col, target='isFraud', n_show=20):
    if 'null' not in col:
        train.fillna(-1, inplace=True)
    tmp1 = train[col].value_counts(dropna=False, normalize=True).reset_index().rename(columns=
                {'index':'feature', col:'freq'})
    if target is not None:
        tmp2 = train[[col, target]].groupby(col)[target].mean().reset_index().rename(columns=
                    {col:'feature', target:'target_mean'})
        tmp = tmp1.merge(tmp2, how='left', on='feature').sort_values(by='freq', ascending=False).reset_index(drop=True).head(n_show)
    else:
        tmp = tmp1.sort_values(by='freq', ascending=False).reset_index(drop=True).head(n_show)
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    ax1.bar(list(tmp.index),tmp['freq'],)
    ax1.set_xticks(tmp.index)
    ax1.set_xticklabels(list(tmp['feature']))
    if target is not None:
        ax2 = ax1.twinx()
        ax2.plot(list(tmp.index), tmp['target_mean'], color='k')
        ax2.set_xticks(tmp.index)
        ax2.set_xticklabels(list(tmp['feature']))
    return tmp

def check_corr(df, feature, n_show=10):
    if feature in cat_used or feature not in df.columns:
        return
    res = dict()
    res['col'] = []
    res['corr'] = []
    for col in df:
        if col != feature and col not in cat_used:
            min_val = np.min([df[feature].min(), df[col].min()])
            res['col'].append(col)
            res['corr'].append(np.abs(np.corrcoef(df[feature].fillna(min_val-1), 
                                                   df[col].fillna(min_val-1))[0][1]))
    res = pd.DataFrame.from_dict(res).sort_values(by='corr', ascending=False)
    return res.head(n_show)
    
IS_DEBUG = False
IS_FREQ_ENCODING = True
IS_AMT_AGG = True
IS_D15_AGG = True
IS_D1_AGG = False
IS_nulls1_AGG = False
IS_dist1_AGG = True
IS_id_02_AGG = False
IS_UID = True
IS_INTERACTION = False
IS_REMOVE_TS = True
output = 'D://kaggle//ieee-fraud-detection//submit//ieee_0826_1'

show_cur_time()
print("loading jupyter ...")
t = time()
proj_path = "D://kaggle//ieee-fraud-detection"
if IS_DEBUG:
    n_rows = 10000
else:
    n_rows = None
    
train_identity= pd.read_csv(os.path.join(proj_path, "input/train_identity.csv"), nrows=n_rows)
train_transaction= pd.read_csv(os.path.join(proj_path, "input/train_transaction.csv"), nrows=n_rows)
test_identity= pd.read_csv(os.path.join(proj_path, "input/test_identity.csv"), nrows=n_rows)
test_transaction = pd.read_csv(os.path.join(proj_path, "input/test_transaction.csv"), nrows=n_rows)
sub = pd.read_csv(os.path.join(proj_path, 'input/sample_submission.csv'), nrows=n_rows)


# perform some pharsing on identity features:
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

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe

train_identity = id_split(train_identity)
test_identity = id_split(test_identity)

# Creat our train & test dataset
train = train_transaction.merge(train_identity, how='left', on='TransactionID')
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

del train_identity,train_transaction,test_identity, test_transaction
gc.collect()
print("loading jupyter finished in {}s".format(time() - t))

show_cur_time()
print("basic feature engineering ...")
t = time()
useful_features = ['TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6',
                   'addr1', 'addr2','dist1','dist2','P_emaildomain','R_emaildomain',
                   'C1','C2','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                   'D1','D2','D3','D4','D5','D6','D8','D9','D10','D11','D12','D13','D15',
                   'M2','M3','M4','M5','M6','M8','M9',
                   'V5','V12','V19','V20','V30','V34','V35','V36','V37','V38','V44','V45','V53',
                   'V54','V55','V56','V61','V62','V66','V67','V70','V74','V75','V76','V77','V78',
                   'V82','V83','V86','V87','V96','V99','V102','V126','V127','V128','V129','V130',
                   'V131','V133','V134','V136','V137','V143','V149','V152','V160','V165','V170',
                   'V171','V187','V189','V203','V204','V207','V209','V210','V217','V222','V233',
                   'V257','V258','V261','V265','V266','V267','V274','V279','V280','V281','V282',
                   'V283','V285','V287','V289','V291','V292','V294','V296','V298','V306','V307',
                   'V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318',
                   'V320','V321','V323','V332',
                   'id_01','id_02','id_05','id_06','id_09','id_13','id_14','id_17','id_18','id_19',
                   'id_20','id_30','id_31','id_32','id_33','id_38','DeviceType','DeviceInfo']

#    time_shift_cols = ['D15', 'D10', 'D4', 'D11', 'D13', 'D6', 'D14'] + ['id_31', 'id_13'] + \
#                        ['M8', 'M7', 'M9', 'M2', 'M3', 'M1'] + ['V81', 'V80', 'V85', 'V84', 
#                        'V93', 'V92', 'V60', 'V59', 'V64', 'V63', 'V18', 'V17', 'V40', 'V72', 
#                        'V39', 'V71', 'V43', 'V21', 'V22', 'V42', 'V31', 'V32', 'V50', 'V91', 
#                        'V90', 'V48', 'V49', 'V70', 'V69', 'V89', 'V52', 'V29', 'V30', 'V61', 
#                        'V94', 'V62', 'V76', 'V51', 'V79', 'V82', 'V83', 'V77', 'V19', 'V66', 
#                        'V20', 'V67', 'V74', 'V45', 'V38', 'V47', 'V75', 'V46', 'V41', 'V37', 
#                        'V35', 'V36', 'V88', 'V34', 'V87', 'V68', 'V53', 'V44', 'V86', 'V73', 
#                        'V2', 'V1', 'V25', 'V8', 'V33', 'V9', 'V7', 'V27', 'V28', 'V4', 'V78', 
#                        'V3', 'V11', 'V57', 'V10', 'V6', 'V5', 'V58', 'V16', 'V56', 'V26', 'V15', 
#                        'V55', 'V12', 'V23', 'V24', 'V65', 'V14', 'V13']
#  time-shifted features to be removed after feature engineering
time_shift_cols = ['D15', 'D10', 'D4', 'D11', 'D13', 'id_31', 'id_13'] + ['V81', 'V80', 'V85', 'V84', 
                 'V93', 'V92', 'V60', 'V59', 'V64', 'V63', 'V18', 'V17', 'V40', 'V72', 'V39', 'V71', 
                 'V43', 'V21', 'V22', 'V42', 'V32', 'V31']
# add pharsed id info
useful_features.extend(['device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id'])

cols_to_drop = [col for col in train.columns if col not in useful_features]
cols_to_drop.remove('isFraud')
cols_to_drop.remove('TransactionID')
cols_to_drop.remove('TransactionDT')
print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

###########################################
############### FE probing part ###########
###########################################
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24


# email binning
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 
          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 
          'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 
          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other',
          'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 
          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 
          'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 
          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 
          'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 
          'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att',
          'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink',
          'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 
          'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 
          'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us') 
    
# fraudulent email domains :https://www.kaggle.com/c/ieee-fraud-detection/discussion/100778
train['P_isproton'] = (train['P_emaildomain'] == 'protonmail.com')
train['R_isproton'] = (train['R_emaildomain'] == 'protonmail.com')
test['P_isproton'] = (test['P_emaildomain'] == 'protonmail.com')
test['R_isproton'] = (test['R_emaildomain'] == 'protonmail.com')

train['P_ismail'] = (train['P_emaildomain'] == 'mail.com')
train['R_ismail'] = (train['R_emaildomain'] == 'mail.com')
test['P_ismail'] = (test['P_emaildomain'] == 'mail.com')
test['R_ismail'] = (test['R_emaildomain'] == 'mail.com')

    
# New feature - decimal part of the transaction amount
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

# the number of null feature per transactionID
#train['nulls1'] = train.isna().sum(axis=1)
#test['nulls1'] = test.isna().sum(axis=1)

# New feature - log of transaction amount. ()
train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

# Count encoding for card1 feature. 
# Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
'''
# removing noise or missing info? It tures out the info loss is more severe
valid_card = train['card1'].value_counts()
valid_card = valid_card[valid_card>10]
valid_card = list(valid_card.index)
    
train['card1'] = np.where(train['card1'].isin(valid_card), train['card1'], np.nan)
test['card1']  = np.where(test['card1'].isin(valid_card), test['card1'], np.nan)
'''
if IS_UID:
    # try userid for further aggregations
    train['card_id'] = train['card1'].astype(str) + '-' + train['card2'].astype(str) + '-' + train['card3'].astype(str) + \
                    '-' + train['card4'].astype(str) + '-' + train['card5'].astype(str) + '-' + train['card6'].astype(str)
    test['card_id'] = test['card1'].astype(str) + '-' + test['card2'].astype(str) + '-' + test['card3'].astype(str) + \
                    '-' + test['card4'].astype(str) + '-' + test['card5'].astype(str) + '-' + test['card6'].astype(str)
                    
    train['uid'] = train['card1'].astype(str) + '-' + train['card2'].astype(str) + '-' + train['card3'].astype(str) + \
                    '-' + train['card4'].astype(str) + '-' + train['card5'].astype(str) + '-' + train['card6'].astype(str) + \
                    '-' + train['addr1'].astype(str)
    test['uid'] = test['card1'].astype(str) + '-' + test['card2'].astype(str) + '-' + test['card3'].astype(str) + \
                    '-' + test['card4'].astype(str) + '-' + test['card5'].astype(str) + '-' + test['card6'].astype(str) + \
                    '-' + test['addr1'].astype(str)

            
if IS_INTERACTION:
    print("build cross feature interaction...")
    # Some arbitrary features interaction
#     cross_features = ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 
#                       'P_emaildomain__C2','card2__dist1', 'card1__card5', 'card2__id_20', 
#                       'card5__P_emaildomain', 'addr1__card1']
#     cross_features = ['addr1__card1', 'addr1__card2', 'addr1__card5', 'addr1__D2', 'addr1__D15', 'addr1__C13', 'addr1__id_02',
#                       'card1__dist1', 'card1__card2', 'card1__card5', 'card1__D2', 'card1__D15', 'card1__C13', 'card1__id_02',
#                       'card2__dist1', 'card2__card2', 'card2__card5', 'card2__D2', 'card2__D15', 'card2__C13', 'card2__id_02',
#                       'P_emaildomain__card1', 'P_emaildomain__card2', 'P_emaildomain__card5', 'P_emaildomain__D2', 'P_emaildomain__D15',
#                       'P_emaildomain__C13', 'P_emaildomain__dist1',  'P_emaildomain__id_02',]
    cross_features = ['addr1__card1', 'addr1__card2','card1__dist1', 'P_emaildomain__card1', 'P_emaildomain__card2']
    for feature in tqdm(cross_features):
        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

if IS_FREQ_ENCODING:
    print("perform freqency encoding...")
    # freq_encode_cols = ['card1', 'card2', 'addr1', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']
    freq_encode_cols_full = ['card1', 'card2', 'card5', 'addr1', 'dist1','P_emaildomain', 'R_emaildomain', 
                             'P_emaildomain_bin', 'R_emaildomain_bin', 
                             'D1', 'D2', 'D4', 'D10', 'D11', 'D15', 'C13',
                             'nulls1', 'id_02', 'Transaction_decimal', 
                             'Transaction_day_of_week', 'Transaction_hour']
    if IS_UID:
        freq_encode_cols_full.extend(['uid', 'card_id'])
        
    if IS_INTERACTION:
        freq_encode_cols_full.extend(cross_features)
    for col in tqdm(freq_encode_cols_full):
        if col in useful_features and col not in time_shift_cols:
            train[col+'_count_full'] = train[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
            test[col+'_count_full'] = test[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))


common_group_cols = ['card1','card2', 'addr1', 'ProductCD', 'Transaction_hour']
#common_group_cols = ['card1','card2','card5', 'addr1', 'ProductCD', 'Transaction_hour']
#common_group_cols = ['card1','card2','card5', 'addr1', 'nulls1', 'Transaction_hour']
#if IS_INTERACTION:
#        common_group_cols += cross_features

# low frequency filtering
print("perform low freq filtering ...")
for col in tqdm(list(test.columns)): 
    valid_col = pd.concat([train[[col]], test[[col]]])
    valid_col = valid_col[col].value_counts()
    # we consider it a common group only if its value show up more than 1 times in whole dataset
    valid_col = valid_col[valid_col>1]
    valid_col = list(valid_col.index)

    train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
    test[col]  = np.where(test[col].isin(train[col]), test[col], np.nan)

    train[col] = np.where(train[col].isin(valid_col), train[col], np.nan)
    test[col]  = np.where(test[col].isin(valid_col), test[col], np.nan)


#agg_types = ['mean', 'median','std', ]
#agg_types = ['mean', 'max','std', ]
agg_types = ['mean', 'std']
if IS_AMT_AGG:
    print("perform group aggregation on Amount...")
    agg_feature = 'TransactionAmt'
    # aggregations on transactionAmt
    #group_cols = ['card1','card2','card3','card5', 'uid', 'card_id'] # card 4,6 have limited unique values
    #group_cols = ['card1','card2','card3','card5', 'addr1']
    group_cols = deepcopy(common_group_cols)
    if IS_UID:
        group_cols.extend(['uid', 'card_id'])
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)
    
if IS_D15_AGG:
    print("perform group aggregation on D15...")
    agg_feature = 'D15'
    # aggregations on transactionAmt
    group_cols = deepcopy(common_group_cols)
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)
if IS_D1_AGG:
    print("perform group aggregation on D1...")
    agg_feature = 'D1'
    # aggregations on transactionAmt
    group_cols = deepcopy(common_group_cols)
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)
if IS_dist1_AGG:
    print("perform group aggregation on dist1...")
    agg_feature = 'dist1'
    # aggregations on transactionAmt
    group_cols = deepcopy(common_group_cols)
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)

if IS_id_02_AGG:
    print("perform group aggregation on id_02...")
    agg_feature = 'id_02'
    # aggregations on transactionAmt
    group_cols = deepcopy(common_group_cols)
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)

if IS_nulls1_AGG:
    print("perform group aggregation on nulls1...")
    agg_feature = 'nulls1'
    # aggregations on transactionAmt
    group_cols = deepcopy(common_group_cols)
    group_cols.remove('nulls1')
    #group_cols = ['card1','card2','card5', 'addr1',]
    for col in tqdm(group_cols):
        for agg_type in agg_types:
            new_col_name = col + '_'+ agg_feature + '_' + agg_type
            temp_df = pd.concat([train[[col, agg_feature]], test[[col,agg_feature]]]).groupby([col])[
                    agg_feature].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_mapping = temp_df[new_col_name].to_dict()   
        
            train[new_col_name] = train[col].map(temp_mapping)
            test[new_col_name]  = test[col].map(temp_mapping)
            
#            train[new_col_name + '_ratio'] = train[agg_feature]/(train[new_col_name] + 1e-6)
#            test[new_col_name + '_ratio'] = test[agg_feature]/(test[new_col_name] + 1e-6)
            
if IS_UID:
    train = train.drop(['card_id', 'uid'], axis=1)
    test = test.drop(['card_id', 'uid'], axis=1)


'''
# remove colinear features : lb drops from 0.9466 to 0.9446 ...
#colinear_cols = ['card1__card5', 'addr1__card1', 'card2__dist1', 'card2__id_20', 'P_emaildomain__C2',
#                 'D1', 'card5__P_emaildomain', 'C14', 'C2', 'C6', 'C11', 'C1', 'DeviceInfo__P_emaildomain',
#                 'id_02__D8']
colinear_cols = ['D1', 'C14', 'C2', 'C6', 'C11', 'C4', 'C8', 'C10', ]
trn_remain_cols = [col for col in train.columns if col not in colinear_cols]
test_remain_cols = [col for col in test.columns if col not in colinear_cols]
train = train[trn_remain_cols]
test = test[test_remain_cols]
    '''

#train_copy = deepcopy(train)
#train_copy.to_csv(os.path.join(proj_path, "input/train_features0815.csv"), index=False)

#features_selected = list(test.columns)
#if IS_REMOVE_TS:
#    # remove time-based features after feature engineering
#    features_selected.remove('D5')
#    features_selected.remove('id_31')
#    features_selected.remove('D15')
#    features_selected.remove('D10')
#    features_selected.remove('D4')
#    features_selected.remove('id_13')
features_selected = list(test.columns)
if IS_REMOVE_TS:
    for col in time_shift_cols:
        if col in features_selected:
            features_selected.remove(col)

categorical_features = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
                        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", 
                        "M5","M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo", "id_12", "id_13", "id_14",
                        "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23",
                        "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", 
                        "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
                        'P_emaildomain_bin', 'P_emaildomain_suffix', 
                        'R_emaildomain_bin', 'R_emaildomain_suffix',
                        'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                        'browser_id_31', 'version_id_31', 'had_id'
                        ]
if IS_INTERACTION:
    categorical_features = list(set(categorical_features).union(set(cross_features)))
cat_used = [col for col in categorical_features if col in train.columns]

print("encode categoricals...")     
for col in tqdm(train.columns):
    if train[col].dtype == 'object' or col in cat_used:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


###########################################
############### experiment part ###########
###########################################
        
X = train[features_selected + ['isFraud']].sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test[features_selected].sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

# del train
# gc.collect()

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }
num_splits =  5
fold_len = int(len(X)/(num_splits+1))
folds = TimeSeriesSplit(n_splits=num_splits, max_train_size=fold_len*3)
#folds =  KFold(n_splits=5, shuffle=False, random_state=2019)

aucs = list()
feature_importances = pd.DataFrame()
oof = np.zeros(len(X))
predictions = np.zeros(len(X_test))
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], categorical_feature="auto", 
                    verbose_eval=100, early_stopping_rounds=300)
    # predict for test set
#    predictions += clf.predict(X_test) / folds.n_splits
    # predict for oof 
    oof_predictions = clf.predict(X.iloc[val_idx])
    oof[val_idx] = oof_predictions
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC by folds:', np.mean(aucs))
# more recent more important, only evaluate on the recent 3 folds
print('Mean AUC by jupyter:', roc_auc_score(y[3*fold_len:], oof[3*fold_len:]))
print('-' * 30)

feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv(output+'_fi.csv', index=False)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));

#sub_cv = deepcopy(sub)
#sub_cv['isFraud'] = predictions
#sub_cv.to_csv(output+'_cv.csv', index=False)

# clf right now is the last model, trained with 80% of jupyter and validated with 20%
best_iter = clf.best_iteration
clf_final = lgb.LGBMClassifier(**params, n_estimators=best_iter)
clf_final.fit(X, y)
sub['isFraud'] = clf_final.predict_proba(X_test)[:, 1]
sub.to_csv(output+'.csv', index=False)




#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([1, 2, 3, 4, 5, 6])
#tscv = TimeSeriesSplit(n_splits=5, max_train_size=3)
#print(tscv)  
#for train_index, test_index in tscv.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
    