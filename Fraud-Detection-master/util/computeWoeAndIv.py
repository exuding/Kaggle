#-*- coding:utf-8 -*-
'''
@project: Fraud-Detection
@author: taoxudong
@time: 2019-09-11 09:59:05 
'''
import pandas as pd
import numpy as np
# 将变量的值转化为相应的分组
def valueToGroup(x, cutoffs):  # 需要转化到分组的值，cutoffs各组的启始值
    # 切分点从小到大排序
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)
    # 异常情况，小于第一组的起始值，这里直接放到第一组
    # 异常值建议在分组之前处理妥善
    if x < cutoffs[0]:
        return 'group1'
    for i in range(1, num_groups):
        if cutoffs[i - 1] <= x < cutoffs[i]:
            return 'group{}'.format(i)
    # 最后一组，也可能会包含很大的异常值
    return 'group{}'.format(num_groups)


# 将变量的值转化为相应的WOE编码
def valueToWoe(x, cutoffs, woe_map):  # 需要转化到分组的值，cutoffs各组的启始值 woe_map:woe编码字典
    # 切分点从小到大排序
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)
    group = None
    # 异常情况，小于第一组的起始值，这里直接放到第一组
    # 异常值建议在分组之前处理妥善
    if x < cutoffs[0]:
        return 'group1'
    for i in range(1, num_groups):
        if cutoffs[i - 1] <= x < cutoffs[i]:
            group = 'group{}'.format(i)
            break
    # 最后一组，也可能会包含很大的异常值
    if group is None:
        group = 'group{}'.format(num_groups)
    if group in woe_map:
        return woe_map[group]
    return None


# 计算WOE编码
def computWoe(df, var, target):  # df:数据集 var:以分组的列名 target：相应变量（0，1）,labels
    eps = 0.000001  # 避免除以0
    gbi = pd.crosstab(df[var], df[target]) + eps
    gb = df[target].value_counts() + eps
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    return gbri['woe'].to_dict()


# 计算IV值
'''
量化指标含义如下：
< 0.02useless for prediction、
0.02 to 0.1Weak predictor、
0.1 to 0.3Medium predictor、
0.3 to 0.5Strong predictor 、
>0.5 Suspicious or too good to be true
'''
def computIV(df, var, target):
    eps = 0.000001  # 避免除以0
    gbi = pd.crosstab(df[var], df[target]) + eps
    gb = df[target].value_counts() + eps
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    gbri['iv'] = (gbri[1] - gbri[0]) * gbri['woe']
    return gbri['iv'].sum()