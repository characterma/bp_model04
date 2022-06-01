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
from config import Min_Minute

from sklearn.ensemble import IsolationForest

"""
根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据,再加上此用户的age, height, weight, gender特征, 把用户自己测量的
sbp, dbp作为label, 构造数据
"""


def find_feature_bp_by_user(wear_user_id, min_minute):

    ansql = AndunSql()
    feature_names_sql = ['Slope','DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime','A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 
                        'A1', 'A2', 'AC']


    feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
                    'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime',
                    'A1', 'A2', 'AC',
                    #  'Age', 'Height', 'Weight', 'AvgSBP','AvgDBP','Gender_0','Gender_1'
                    ]


    def time_stamp(x):
        return "{} {}:{}:{}".format(str(x[0]), x[1][0:2], x[1][2:4], x[1][4:6])

    ppg_history_path = "personal_models/data/{}_ppg_history.csv".format(wear_user_id)
    select_user_path = "personal_models/data/{}_kefu_history.csv".format(wear_user_id)

    select_user = pd.read_csv(select_user_path, dtype={'wear_user_id':'object'})
    select_user['gmt_create'] = select_user['gmt_create'].apply(lambda x: pd.to_datetime(x))
    select_user['wear_user_id'] = select_user['wear_user_id'].apply(lambda x: str(x))
    # select_user['gmt_create'] = select_user['gmt_create'].apply(lambda x: x.value // 10**9)
    print(select_user.head())
    print(select_user.dtypes)


    ppg_history = pd.read_csv(ppg_history_path, dtype={'wear_user_id':'object'})
    # ppg_history['wear_user_id'] = ppg_history['wear_user_id'].apply(lambda x: str(x))
    ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: str(x).rjust(6, '0'))
    ppg_history['ppg_time'] = ppg_history[['date', 'ppg_time']].apply(lambda x: time_stamp(x), axis=1)
    ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: pd.to_datetime(x))
    ppg_history['wear_user_id'] = ppg_history['wear_user_id'].apply(lambda x: str(x))
    # ppg_history['ppg_time'] = ppg_history['ppg_time'].apply(lambda x: x.value // 10**9)

    print(ppg_history.head())
    print(ppg_history.dtypes)


    # 保存最终的结果
    # results = []
    results = pd.DataFrame(columns=feature_names + ['wear_user_id', 'gmt_create', 'ppg_time', 'sbp', 'dbp', 'age', 'height', 'weight', 'gender'])

    # 查询 此用户的基本信息
    age = ansql.ansql_user_age(wear_user_id)
    gender = ansql.ansql_user_gender(wear_user_id)
    height = ansql.ansql_user_height(wear_user_id)
    weight = ansql.ansql_user_weight(wear_user_id)

    # 从 客服记录中找出 此用户 的所有的 血压记录
    temp_select_user = select_user[select_user['wear_user_id'] == wear_user_id]
    # 从 ppg_history 里面找出此用户的 这几天的 ppg 数据
    temp_ppg_history = ppg_history[ppg_history['wear_user_id'] == wear_user_id]

    if temp_select_user.shape[0] > 0 and temp_ppg_history.shape[0] > 0:
        # 遍历 每个 客服记录的数据, 找出时间上最接近 的 ppg信号的数据
        for a in range(temp_select_user.shape[0]):
        
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
                
                # print(diff_temp_ppg_history['diff_time_seconds'].values)
                min_index = pd.Series(diff_temp_ppg_history['diff_time_seconds'].values).argmin()
                for min_index in range(diff_temp_ppg_history.shape[0]):
                    # print(min_index)
                    ppg_values = diff_temp_ppg_history.iloc[min_index]['ppg_values']
                    ppg_values = list(ppg_values.split(','))
                    ppg_values = np.array(ppg_values)
                    ppg_values = ppg_values.reshape(-1, 12)

                    # 如果此 条 ppg_value 组数小于3 ,则跳过
                    if ppg_values.shape[0] < 3:
                        continue

                    # 把ppg_values保存成 dataframe
                    temp_result = pd.DataFrame(data=ppg_values, columns=feature_names_sql)

                    temp_result['wear_user_id'] = wear_user_id
                    temp_result['gmt_create'] = user_bp_time
                    temp_result['ppg_time'] = diff_temp_ppg_history.iloc[min_index]['ppg_time']
                    
                    temp_result['sbp'] = temp_select_user.iloc[a]['sbp']
                    temp_result['dbp'] = temp_select_user.iloc[a]['dbp']
                    temp_result['age'] = age
                    temp_result['height'] = height
                    temp_result['weight'] = weight
                    temp_result['gender'] = gender

                    # results.append(temp_result)
                    # 拼接到 df_results
                    results = pd.concat([results, temp_result], ignore_index=True)

                    # print(temp_ppg_history.head())
            else:
                print("{} - {} 没有 小于{} min 的ppg数据...".format(wear_user_id, user_bp_time, min_minute))

                # 把后台客服记录的此条数据, 状态置为1, 表示此条数据删除
                # ansql.update_kefu_bp_history(wear_user_id, user_bp_time)

    else:
        print("{} 的数据不全...".format(wear_user_id))

    feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
                    'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime', 
                    'A1', 'A2', 'AC']


    # print("开始对 SBP 进行异常值检测...")
    # if_model = IsolationForest(n_estimators=80)
    # pred = if_model.fit_predict(results[feature_names])

    # results['if_pred'] = pred

    # 2020-09-21计算几个新特征
    # results['AC'] = results['AC'].astype('float32')
    # results['SystolicTime'] = results['SystolicTime'].astype('float32')
    # results['DiastolicTime'] = results['DiastolicTime'].astype('float32')
    results = results.astype({
                            'A1/AC': 'float32',
                            'A2/AC': 'float32',
                            'Slope': 'float32',
                            'DiastolicTime': 'float32',
                            'SystolicTime': 'float32',
                            'RR': 'float32',
                            'UpToMaxSlopeTime': 'float32',
                            'A1': 'float32',
                            'A2': 'float32',
                            'AC': 'float32',
                            'sbp': 'float32',
                            'dbp': 'float32'
                            })
    
    # results['Volume'] = results[['AC', 'SystolicTime', 'DiastolicTime']].apply(lambda x: float(x['AC']) * (1.0 + float(x['SystolicTime']) / float(x['DiastolicTime'])))
    results['Volume'] = results.apply(lambda x: float(x['AC']) * (1.0 + float(x['SystolicTime']) / float(x['DiastolicTime'])), axis=1)
    results['Volume/Slope'] = results['Volume'] / results['Slope']
    results['Volume/RR'] = results['Volume'] / results['RR']

    
    # 相关性分析
    # print(results.corr())


    results.to_csv("personal_models/data/{}_ppg_data.csv".format(wear_user_id), index=False)

    # 先对 results 的时间, 进行去重
    results.drop_duplicates(subset=['gmt_create','sbp'], keep='first', ignore_index=True,  inplace=True)
    # 返回 血压记录的总 条数
    # un_sbp = results['sbp'].unique()
    # un_dbp = results['dbp'].unique()
    un_sbp = results['sbp'].tolist()
    un_dbp = results['dbp'].tolist()


    return un_sbp, un_dbp



if __name__ == "__main__":

    wear_user_id = "a59cacfa"
    # 设置 时间差 3min, 用于寻找 用户自己测量血压的时间和 ppg信号时间的匹配, 大于此时间差的直接排除, 从剩余的里面找出时间差最小的
    min_minute = Min_Minute

    find_feature_bp_by_user(wear_user_id=wear_user_id, min_minute=min_minute)