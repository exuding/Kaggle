#-*- coding:utf-8 -*-
'''
@project: Fraud-Detection
@author: taoxudong
@time: 2019-09-19 11:31:34 
'''

from util.splitBox import ChiMerge
from util.computeWoeAndIv import valueToGroup
from util.computeWoeAndIv import computWoe,computIV
import os
import pickle
import multiprocessing
from util.reduceMemory import reduce_mem_usage
import pandas as pd
import gc

#压缩数据
def load_data(file):
    return reduce_mem_usage(pd.read_csv(file))


if __name__=='__main__':
    files = ['./input/test_identity.csv',
             './input/test_transaction.csv',
             './input/train_identity.csv',
             './input/train_transaction.csv']
    # 线程池读取多个数据集
    with multiprocessing.Pool() as pool:
        test_identity, test_transaction, train_identity, train_transaction = pool.map(load_data,files)
    # 合并数据
    train_re = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    test_re = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
    del test_identity, test_transaction, train_identity, train_transaction
    # 垃圾回收机制
    gc.collect()


    # 读取数据
    pickle_file = 'trainAndTestData0.9476.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  # 反序列化，与pickle.dump相反
        train = pickle_data['train_dataset']
        y_train = pickle_data['train_labels']
        sample_submission = pickle_data['valid_dataset']
        del pickle_data  # 释放内存

    train['isFraud']= train_re['isFraud']
    #连续型变量先分箱后计算iv值
    feature = [x for x in train.columns]

    featureImport = {}
    for item in feature:
        train[item] = train[item].fillna(-999)
        # 卡方分箱法对变量进行分箱 得到分箱点
        cutoffs = ChiMerge(train, item, 'isFraud', max_interval=6)
        train[item + '_group'] = train[item].apply(valueToGroup, args=(cutoffs,))
        # woe和iv后面用到再释放出来
        woe_map = computWoe(train,item+'_group','isFraud')
        print('woe_map{}'.format(item),woe_map)
        iv = computIV(train,item+'_group','isFraud')
        featureImport[item+'_group_featureImportant'] = iv

    #离散变量直接计算Iv值

    # 保存数据方便调用
    pickle_file = 'trainFeatureImportant.pickle'
    if not os.path.isfile(pickle_file):  # 判断是否存在此文件，若无则存储
        print('Saving jupyter to pickle file...')
        try:
            with open('trainFeatureImportant.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'featureImport': featureImport,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save jupyter to', pickle_file, ':', e)
            raise
