#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
np.random.seed(666)

from tqdm import tqdm
import sys
sys.path.append(r"utils/")
from config import feature_names, feature_names_sql

"""
把 results_3, results_5, results_7 首先split 为  train_data  和  test_data,
然后再提取每条数据里面的feature, 整理成训练数据格式
"""


# 数据 加载与处理
raw_data_path = r"data/raw_data/data_kefu/results_8.csv"
# raw_data_path = r"data/data_from_sql/data/results_5.csv"

raw_data = pd.read_csv(raw_data_path, encoding='utf-8')

# shuffle
raw_data = shuffle(raw_data)
# 增加 id 字段,用于区分 train, test
raw_data['id'] = list(range(raw_data.shape[0]))
# 用于 提取夜晚的ppg信号特征
raw_data['time'] = raw_data['ppg_time'].apply(lambda x: int(x[-8:].replace(":", '')))
print(raw_data.head())
print(raw_data.dtypes)



# 开始 提取ppg字段的feature
df_results = pd.DataFrame(columns=feature_names + ['wear_user_id', 'time'])

# 遍历 raw_data 里面的每条数据, 处理 ppg_values 字段
for a in tqdm(range(raw_data.shape[0])):
    ppg_values = raw_data.iloc[a]['ppg_values']
    ppg_values = list(ppg_values.split(','))
    ppg_values = np.array(ppg_values)
    ppg_values = ppg_values.reshape(-1, 12)

    # 组合dataframe
    df_feature = pd.DataFrame(data=ppg_values, columns=feature_names_sql)
   
    df_feature['Age'] = raw_data.iloc[a]['age']
    df_feature['Height'] = raw_data.iloc[a]['height']
    df_feature['Weight'] = raw_data.iloc[a]['weight']
    df_feature['AvgSBP'] = raw_data.iloc[a]['sbp']
    df_feature['AvgDBP'] = raw_data.iloc[a]['dbp']
    df_feature['id'] = raw_data.iloc[a]['id']
    df_feature['time'] = raw_data.iloc[a]['time']

    # 把每条数据添加 wear_user_id 用于后面的比较误差
    df_feature['wear_user_id'] = raw_data.iloc[a]['wear_user_id']

    
    # assert(df_feature.shape[1] == 15)
 
    gender = raw_data.iloc[a]['gender']
    if gender == 1:
        df_feature['Gender_0'] = 0
        df_feature['Gender_1'] = 1
    else:
        df_feature['Gender_0'] = 1
        df_feature['Gender_1'] = 0

    # 按照 xgboost, random forest训练数据的特征顺序
    df_feature = df_feature[feature_names + ['id','wear_user_id', 'time']]

    # 拼接到 df_results
    df_results = pd.concat([df_results, df_feature], ignore_index=True)

print("共有数据条: {}".format(df_results.shape[0]))


# 更改数据类型
df_results = df_results.astype({'A1':'float', 'A2':'float', 'AC':'float'})

# 因为 数据库里面的 PPG信号数据 A1/AC等4个比值的特征跟实际计算不一样, 所以这里重新手动计算一下,替换掉原来的值
df_results['A1/(A1+A2)'] = df_results[['A1', 'A2']].apply(lambda x: x[0] / (x[0] + x[1]), axis=1)
df_results['A2/(A1+A2)'] = df_results[['A1', 'A2']].apply(lambda x: x[1] / (x[0] + x[1]), axis=1)

df_results['A1/AC'] = df_results[['A1', 'AC']].apply(lambda x: x[0] / x[1], axis=1)
df_results['A2/AC'] = df_results[['A2', 'AC']].apply(lambda x: x[0] / x[1], axis=1)

print("df_results.shape:", df_results.shape)
# print(df_results.head())

"""
找出夜晚 2点至5点的数据, 统计出 mean, std 值,用于过滤数据
"""
night_data = df_results[df_results['time'] <=50000]
night_data = night_data[night_data['time'] >=10000]
print("夜间的数据一共有 {} 条....".format(night_data.shape[0]))



# 对数据进行统计分析,去除 15%, 85%的异常值
# df_data[feature_names].astype(np.float32).describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
features_statistics = night_data[feature_names].describe(percentiles=[0.15,0.30,0.5,0.70,0.85])
print(features_statistics)

# 按照上面 15%, 85%分位值,把异常值删除
# for fsc in features_statistics.columns:
# for fsc in ['A1', 'A2', 'AC']:
for fsc in ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC','A1', 'A2', 'AC']:
    if fsc not in ['id', 'Gender_0','Gender_1', 'Age', 'Height', 'Weight', 'AvgSBP','AvgDBP']:
        print("feature_name:", fsc)
        # 删除 小于 15% 的值
        df_results = df_results[df_results[fsc] >= features_statistics.loc['15%', fsc]]
        # 删除 大于 85%的值
        df_results = df_results[df_results[fsc] <= features_statistics.loc['85%', fsc]]

print("按照统计分析, 去除 15%, 85%的极端值后,剩余数据 {}".format(df_results.shape[0]))

print('重新统计:')
print(df_results[feature_names].describe(percentiles=[0.15,0.30,0.5,0.70,0.85]))



"""
统计血压区间占比
"""
bp_zones = list(range(50, 210, 10))
bp_zones.append(1000)
bp_labels = ["{}_{}".format(bp_zones[bz], bp_zones[bz+1]) for bz in range(len(bp_zones) - 1)]
# bp_labels.append("{}以上".format(bp_zones[-1]))

df_results['sbp_zone'] = pd.cut(df_results['AvgSBP'], bins=bp_zones, labels=bp_labels, include_lowest=True)

# print(data.groupby(['sbp_zone'])['Slope','DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime'].agg(['mean','std','sum','count']))
print(df_results.groupby(['sbp_zone']).agg(['count']))

df_results.drop(columns=['sbp_zone'], axis=1, inplace=True)




# 划分 train ,test
split_len = int(0.8 * raw_data.shape[0])

data_train = df_results[df_results['id'] < split_len]
print("data_train.shape", data_train.shape)
data_train.to_csv(r"data/train_data/train_data_kefu.csv", index=False)
data_test = df_results[df_results['id'] >= split_len]
print("data_test.shape", data_test.shape)
data_test.to_csv(r"data/test_data/test_data_kefu.csv", index=False)