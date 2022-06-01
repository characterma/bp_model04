#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append(r"utils/")
import os
import json
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
from andun_sql.andun_sql import AndunSql
from utils import cal_time_delta
from tqdm import tqdm
from config import Max_SBP, Min_SBP, Max_DBP, Min_DBP
from config import SIGMA



"""
1 处理去年公司内部采集的数据, 全部使用
首先按照时间 feature 的时间点 进行 train test 划分, 然后 再提取里面的feature
"""


# csv 文件数据的目录
data_path = r"data_process/old_data_process/1_concat_data.csv"

# feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
#        'DiastolicTime', 'SystolicTime', 'RR', 'Age', 'Gender', 'SBP', 'DBP',
#        'Height', 'Weight', 'A1', 'A2', 'AC', 'SBP2', 'DBP2', 
#        'RefDeviceID', 'AvgSBP', 'AvgDBP', 'DiffSBP', 'DiffDBP',
#        'UpToMaxSlopeTime', 'HighBloodPressure', 'HRa', 'HRb','DiffHR'
#        ]

feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
       'DiastolicTime', 'SystolicTime', 'UpToMaxSlopeTime', 'RR', 'A1', 'A2', 'AC'
       ]
# 'WearUserID', 'MedicalHistory',
# 读取数据
df_data = pd.read_csv(data_path)


"""
对数据进行清洗
"""
# 提取出hour 小时
df_data['hour'] = df_data['createTime'].apply(lambda x: x.split(" ")[1][:2])

# 去除 SBP, SBP2 异常高的数据
df_data = df_data[df_data['SBP'] <= Max_SBP]
df_data = df_data[df_data['SBP2'] <= Max_SBP]

df_data = df_data[df_data['SBP'] >= Min_SBP]
df_data = df_data[df_data['SBP2'] >= Min_SBP]

print("去除高压异常值之后,剩余数据 {} ".format(df_data.shape[0]))
# 去除 DBP, DBP2 异常高的数据
df_data = df_data[df_data['DBP'] <= Max_DBP]
df_data = df_data[df_data['DBP2'] <= Max_DBP]

df_data = df_data[df_data['DBP'] >= Min_DBP]
df_data = df_data[df_data['DBP2'] >= Min_DBP]

print("去除低压异常值之后,剩余数据 {} ".format(df_data.shape[0]))


"""
找出此部分数据的 夜晚的数据, 通过夜晚的数据统计 mean, std, 然后进行过滤
"""
night_data = df_data[df_data['Time'] <=50000]
night_data = night_data[night_data['Time'] >=10000]
print("夜间的数据一共有 {} 条....".format(night_data.shape[0]))


# 查询统计结果
# night_data[feature_names].astype(np.float32).describe(percentiles=[0.15,0.5,0.75,0.85])
features_statistics = night_data[feature_names].describe(percentiles=[0.15,0.5,0.75,0.85])
features_statistics = night_data[feature_names].describe(percentiles=[0.01,0.05,0.1,0.15,0.25,0.33,0.5,0.68,0.75,0.85,0.9,0.95,0.99])
# 需要保存 index ,后面用于过滤数据
features_statistics.to_csv("data/old_data_bp_features_analysis.csv")
print(features_statistics)

# 按照 mean, std 设置的分位值,把异常值删除
# for fsc in features_statistics.columns:
for fsc in ['A1', 'A2', 'AC']:
    
    # 方法一 按照 mean, std 进行清洗
    # temp_mean = features_statistics.loc['mean', fsc]
    # temp_std = features_statistics.loc['std', fsc]
    # df_data = df_data[df_data[fsc] >= (temp_mean - SIGMA * temp_std)]
    # df_data = df_data[df_data[fsc] <= (temp_mean + SIGMA * temp_std)]

    # 方法二  按照 设置的百分比进行清洗
    # 删除 小于 5% 的值
    df_data = df_data[df_data[fsc] >= features_statistics.loc['15%', fsc]]
    # 删除 大于 95%的值
    df_data = df_data[df_data[fsc] <= features_statistics.loc['85%', fsc]]



print("按照统计分析, 去除 mean, std 设置的至信区间的极端值后,剩余数据 {}".format(df_data.shape[0]))

print('重新统计:')
print(df_data[feature_names].describe(percentiles=[0.15,0.5,0.75,0.85]))





"""
统计血压区间占比
"""
bp_zones = list(range(50, 210, 10))
bp_zones.append(1000)
bp_labels = ["{}_{}".format(bp_zones[bz], bp_zones[bz+1]) for bz in range(len(bp_zones) - 1)]
# bp_labels.append("{}以上".format(bp_zones[-1]))

df_data['sbp_zone'] = pd.cut(df_data['AvgSBP'], bins=bp_zones, labels=bp_labels, include_lowest=True)

# print(data.groupby(['sbp_zone'])['Slope','DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime'].agg(['mean','std','sum','count']))
print(df_data.groupby(['sbp_zone']).agg(['count']))

df_data.drop(columns=['sbp_zone'], axis=1, inplace=True)




"""
进行 train test 划分
"""
file_name_lists = df_data['file_name'].unique()

length = len(file_name_lists)
random.shuffle(file_name_lists)
# 按照 8:2 进行split
train_file_names = file_name_lists[:int(length * 0.8)]
test_file_names = file_name_lists[int(length * 0.8):]

train_data = df_data[df_data['file_name'].isin(train_file_names)]
test_data = df_data[df_data['file_name'].isin(test_file_names)]


# 保存
train_data.to_csv(r"data/train_data/train_data_old.csv", index=False, encoding='utf-8')
test_data.to_csv(r"data/test_data/test_data_old.csv", index=False, encoding='utf-8')

print('df_data.shape: {}'.format(df_data.shape))
print('train_data.shape: {}'.format(train_data.shape))
print('test_data.shape: {}'.format(test_data.shape))