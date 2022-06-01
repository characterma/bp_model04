#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append(r"utils/")
import os
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from andun_sql.andun_sql import AndunSql
from utils import add_new_features, clean_data_with_quantile_statistics, clean_data_with_mean_std
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from tqdm import tqdm


"""
从数据库中 找出找出此用户的所有的bp_feature记录, 并分别按照 夜晚, 白天, 统计bp_feature每个特征的 统计参数, 用于用户数据的过滤
此统计参数也会用于 训练数据, 及 将来线上实时预测时对数据的过滤
"""



def pull_bp_features_by_user_from_db(wear_user_id, last_n_day=None):
    # last_n_day 表示从当前时刻起,往前推 几个月份, 就是对于某个用户只使用最近 若干天的ppg_feature数据做统计分析
    ansql = AndunSql()
    end_date = datetime.now()
    # 数据库中,此用户的第一次上传数据的时间
    start_date = ansql.ansql_user_start_wear_date(wear_user_id)
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # config设置的 此用户的初始 ppg_feature 日期
    if last_n_day:
        start_date_config = end_date - timedelta(days=last_n_day)
        if start_date_config >= start_date:
            start_date = start_date_config

    end_date = end_date + timedelta(days=1)
    end_date = datetime.strftime(end_date, '%Y-%m-%d')
    set_time = pd.date_range(start=start_date, end=end_date)
    set_time = [datetime.strftime(st, '%Y-%m-%d') for st in set_time]
    # print(set_time)
    ppg_history = []

    # 遍历 unique_days
    for ud in set_time:
        print("{} - {}".format(wear_user_id, ud))        
        # temp_ud = uu_bp_feature[uu_bp_feature['DATE'] == datetime.date(datetime.strptime(str(ud), '%Y-%m-%d'))]
        temp_ud = ansql.ansql_bp_feature(wear_user_id, [ud])
        if isinstance(temp_ud, pd.DataFrame) and temp_ud.shape[0] > 0:
        
            # 把 当天的  ppg信号解析
            ppg_list = temp_ud['FROMPPG'].values[0]
            
            for ps in ppg_list.split(";"):
                if len(ps) == 0:
                    continue
                try:
                    temp_ppg_history = []

                    ppg_time, ppg_values = ps.split("/", 1)
                    # ppg_values = ps.split("/")[1]
                    ppg_values = list(ppg_values.split(","))
                    if len(ppg_values) % 12 == 0:
                        temp_ppg_history.append(wear_user_id)
                        temp_ppg_history.append(ppg_time)
                        temp_ppg_history.append(",".join(ppg_values))
                        temp_ppg_history.append(ud)

                        ppg_history.append(temp_ppg_history)
                    else:
                        print("此用户{}当天{}ppg数据长度有误...".format(wear_user_id, ud))
                
                except Exception as e:
                    print(e)
                    # print("ppg_list:", ppg_list)
                    continue
        else:
            print("此用户{}当天{}无ppg数据...".format(wear_user_id, ud))

        time.sleep(0.5)

    # break
    # time.sleep(0.6)


    # 把 ppg_history整理成 dataframe
    df_ppg_history = pd.DataFrame(data=ppg_history, columns=['wear_user_id', 'ppg_time', 'ppg_values', 'date'])
    df_ppg_history.to_csv("personal_models/data/{}_ppg_history_split.csv".format(wear_user_id), index=False)
    print(df_ppg_history.shape)
    print(df_ppg_history.head())



"""
对数据进行统计分析
"""

def bp_features_analysis(wear_user_id):

    user_bp_features_data = pd.read_csv("personal_models/data/{}_ppg_history_split.csv".format(wear_user_id))

    feature_names_new = FEATURE_NAME_NEW
    df_results = pd.DataFrame(columns = feature_names_new + ['date', 'ppg_time']) 
    # 遍历 每个 ppg_value数据
    for a in tqdm(range(user_bp_features_data.shape[0])):

        ppg_values = user_bp_features_data.iloc[a]['ppg_values']
        ppg_values = list(ppg_values.split(','))
        ppg_values = np.array(ppg_values)
        ppg_values = ppg_values.reshape(-1, 12)

        # 把ppg_values保存成 dataframe
        temp_result = pd.DataFrame(data=ppg_values, columns=feature_names_sql)
        temp_result = temp_result.astype('float')
        # 增加新特征
        temp_result = add_new_features(temp_result)
        # 按照训练数据的格式整理字段
        temp_result = temp_result[feature_names_new]

        # 把 UpToMaxSlopeTime 为 0 的值去掉
        # print(temp_result.shape)
        temp_result = temp_result[temp_result['UpToMaxSlopeTime'] > 0]
        # print(temp_result.shape)

        # 增加其他字段
        temp_result['date'] = user_bp_features_data.iloc[a]['date']
        temp_result['ppg_time'] = user_bp_features_data.iloc[a]['ppg_time']

        # 拼接到 df_results
        df_results = pd.concat([df_results, temp_result], ignore_index=True)

    # 开始进行统计
    # df_results = df_results[feature_names_new].astype('float')
    features_statistics = df_results[feature_names_new].astype('float').describe(percentiles=[0.01,0.05,0.1,0.15,0.25,0.33,0.5,0.68,0.75,0.85,0.9,0.95,0.99])
    print(features_statistics)
    # 需要保存 index ,后面用于过滤数据
    features_statistics.to_csv("personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id))

    # 统计所有数据的特征按照 5%~95%的特征过滤, 从 126009,降低到 70272
    # 统计所有数据的特征按照 1%~99%的特征过滤, 从 126009,降低到 111286

    # 统计所有数据的特征按照 mean的 2*std 的特征过滤, 从 126009,降低到 99549
    # 夜晚数据: 48031, 白天数据: 77978
    # 夜晚数据: 37517, 白天数据: 62032, 夜晚删除的比例: 0.2189002935604089, 白天删除的比例: 0.20449357511092872

    # 统计所有数据的特征按照 mean的 2.5*std 的特征过滤, 从 126009,降低到 113665
    
    # 统计所有数据的特征按照 mean的 3*std 的特征过滤, 从 126009,降低到 119259
    # 夜晚数据: 48031, 白天数据: 77978
    # 夜晚数据: 45623, 白天数据: 73636, 夜晚删除的比例: 0.050134288272157566, 白天删除的比例: 0.05568237195106312

    # 统计一下过滤后数据有多少
    night_len_old = df_results[df_results['ppg_time'] <= 50000 ].shape[0]
    day_len_old = df_results[df_results['ppg_time'] > 50000 ].shape[0]
    print("按照统计结果过滤数据前, 数据条数: {}, 夜晚数据: {}, 白天数据: {}".format(df_results.shape, night_len_old, day_len_old))
    # df_results = clean_data_with_quantile_statistics(df_results, features_statistics)
    df_results = clean_data_with_mean_std(df_results, features_statistics)

    night_len_new = df_results[df_results['ppg_time'] <= 50000 ].shape[0]
    day_len_new = df_results[df_results['ppg_time'] > 50000 ].shape[0]
    print("按照统计结果过滤数据后, 数据条数: {}, 夜晚数据: {}, 白天数据: {}, 夜晚删除的比例: {}, 白天删除的比例: {}".format(df_results.shape, night_len_new, day_len_new, (night_len_old-night_len_new)/(night_len_old+1), (day_len_old-day_len_new)/(day_len_old+1)))


if __name__ == "__main__":

    wear_user_id = "c668572d"

    # 从数据库中拉取此用户的所有 bp_features 数据
    pull_bp_features_by_user_from_db(wear_user_id=wear_user_id, last_n_day=30)

    # 对此数据进行统计分析
    # bp_features_analysis(wear_user_id=wear_user_id)
