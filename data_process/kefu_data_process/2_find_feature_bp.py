#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append(r"utils/")
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from andun_sql.andun_sql import AndunSql
from utils import cal_time_delta
from tqdm import tqdm

"""
根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据,再加上此用户的age, height, weight, gender特征, 把用户自己测量的
sbp, dbp作为label, 构造数据
"""
# 设置 时间差 3min, 用于寻找 用户自己测量血压的时间和 ppg信号时间的匹配, 大于此时间差的直接排除, 从剩余的里面找出时间差最小的
min_minute = 5
ansql = AndunSql()

def time_stamp(x):
    return "{} {}:{}:{}".format(str(x[0]), x[1][0:2], x[1][2:4], x[1][4:6])

ppg_history_path = r"data/raw_data/data_kefu/kefu_ppg_history.csv"
select_user_path = r"data/raw_data/data_kefu/select_users.csv"

ppg_history = pd.read_csv(ppg_history_path)
ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: str(x).rjust(6, '0'))
ppg_history['ppg_time'] = ppg_history[['date', 'ppg_time']].apply(lambda x: time_stamp(x), axis=1)
ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: pd.to_datetime(x))
# ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: x.value // 10**9)

print(ppg_history.head())
print(ppg_history.dtypes)


select_user = pd.read_csv(select_user_path)
select_user['gmt_create'] = select_user['gmt_create'].apply(lambda x: pd.to_datetime(x))
# select_user['gmt_create'] = select_user['gmt_create'].apply(lambda x: x.value // 10**9)
print(select_user.head())
print(select_user.dtypes)

# 取出所有的 users
unique_users = select_user['wear_user_id'].unique()
# 遍历 users

# 保存最终的结果
results = []

for uu in tqdm(unique_users):

    # 查询 此用户的基本信息
    age = ansql.ansql_user_age(uu)
    gender = ansql.ansql_user_gender(uu)
    height = ansql.ansql_user_height(uu)
    weight = ansql.ansql_user_weight(uu)


    # 从 客服记录中找出 此用户 的所有的 血压记录
    temp_select_user = select_user[select_user['wear_user_id'] == uu]
    # 从 ppg_history 里面找出此用户的 这几天的 ppg 数据
    temp_ppg_history = ppg_history[ppg_history['wear_user_id'] == uu]

    if temp_select_user.shape[0] > 0 and temp_ppg_history.shape[0] > 0:
        # 遍历 每个 客服记录的数据, 找出时间上最接近 的 ppg信号的数据
        for a in range(temp_select_user.shape[0]):
            # 中间结果
            temp_result = []

            user_bp_time = temp_select_user.iloc[a]['gmt_create']
            temp_ppg_history['diff_time'] = temp_ppg_history['ppg_time'].apply(lambda x: cal_time_delta(x, user_bp_time))
            # temp_ppg_history['diff_time'] = temp_ppg_history['diff_time'].map(lambda x: x / np.timedelta64(1,'m'))
            # temp_ppg_history['diff_time'] = temp_ppg_history['diff_time'].dt.total_seconds()
            temp_ppg_history['diff_time_days'] = temp_ppg_history['diff_time'].apply(lambda x: x.days)
            temp_ppg_history['diff_time_seconds'] = temp_ppg_history['diff_time'].apply(lambda x: x.seconds)

            # 从里面找出 时间差最小的ppg信号
            diff_temp_ppg_history = temp_ppg_history[temp_ppg_history['diff_time_days'].isin([0, -1])]
            diff_temp_ppg_history = diff_temp_ppg_history[diff_temp_ppg_history['diff_time_seconds'] <= min_minute * 60]
            # 判断 是否有剩余的数据
            if diff_temp_ppg_history.shape[0] > 0:
                
                print(diff_temp_ppg_history['diff_time_seconds'].values)
                min_index = pd.Series(diff_temp_ppg_history['diff_time_seconds'].values).argmin()

                temp_result.append(uu)
                temp_result.append(user_bp_time)
                temp_result.append(diff_temp_ppg_history.iloc[min_index]['ppg_time'])
                temp_result.append(diff_temp_ppg_history.iloc[min_index]['ppg_values'])
                temp_result.append(temp_select_user.iloc[a]['sbp'])
                temp_result.append(temp_select_user.iloc[a]['dbp'])
                temp_result.append(age)
                temp_result.append(height)
                temp_result.append(weight)
                temp_result.append(gender)

                results.append(temp_result)

                # print(temp_ppg_history.head())
            else:
                print("{} - {} 没有 小于{} min 的ppg数据...".format(uu, user_bp_time, min_minute))

    else:
        print("{} 的数据不全...".format(uu))


# 把results建立dataframe
results = pd.DataFrame(data=results, columns=['wear_user_id', 'gmt_create', 'ppg_time', 'ppg_values', 'sbp', 'dbp', 'age', 'height', 'weight', 'gender'])
results.to_csv("data/raw_data/data_kefu/results_{}.csv".format(min_minute), index=False)