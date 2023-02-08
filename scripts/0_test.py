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





ansql = AndunSql()


##########################################
########## 测试 一
##########################################
# 按照指定 wear_user_id 和 日期 查找用户的 ppg 数据
# wear_user_id = "6528ba5b"
# # temp_ud = ansql.ansql_bp_feature('9fc16acd', ['2020-08-01'])
# # print(temp_ud['FROMPPG'].values)
# sel_date = '2020-09-22'
# # sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, sel_date)
# sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, sel_date)

# # res_1 = ansql.ansql_read_mysql(sql_s)
# # res_2 = ansql.ansql_read_mysql(sql_s_t)
# # res_1 = ansql.ansql_read_mysql_test_db(sql_s)
# res_2 = ansql.ansql_read_mysql(sql_s)
# # print(res_1)
# print(res_2)



##########################################
########## 测试 二
##########################################
# # 加载排除的用户
# reject_users = pd.read_csv(r"scripts/公司内部人员名单.csv")
# reject_users = reject_users['wear_user_id'].unique()

# # 导出 新血压用户 信息
# sql_users = "SELECT wear_user_id, create_time FROM andun_watch.d_users_bp_model"
# new_model_users = ansql.ansql_read_mysql(sql_users)
# # 排除
# new_model_users = new_model_users[~new_model_users['wear_user_id'].isin(reject_users)]
# unique_users = new_model_users['wear_user_id'].unique()
# # 查找用户基本信息
# if len(unique_users) == 0:
#     print("没有用户......")
# else:
#     if len(unique_users) == 1:
#         sql_user_info  = "SELECT ID,USERNAME,PHONE_NUM,EMERGENCY_PEOPLE,EMERGENCY_PHONE FROM andun_app.a_wear_user WHERE ID = '{}' ".format(unique_users[0])
#     else:
#         sql_user_info  = "SELECT ID,USERNAME,PHONE_NUM,EMERGENCY_PEOPLE,EMERGENCY_PHONE FROM andun_app.a_wear_user WHERE ID in {} ".format(tuple(unique_users))


# user_info = ansql.ansql_read_mysql(sql_user_info)
# print(user_info.head())

# # 进行merge
# new_model_users = pd.merge(new_model_users, user_info, left_on='wear_user_id', right_on='ID', how='left')
# new_model_users.rename(columns={'create_time':'使用新模型时间', 'PHONE_NUM': '用户电话', 'EMERGENCY_PEOPLE': '紧急联系人', 'EMERGENCY_PHONE': '紧急联系人电话'}, inplace=True)

# new_model_users.sort_values(by=['使用新模型时间'], ascending=True, inplace=True)
# print(new_model_users.head())
# new_model_users.to_csv(r"scripts/new_model_users_info.csv", encoding='utf-8', index=False)




################################################
################################################
#  查询指定时间段 的 指定 wear_user_id 的用户 的ppg数据

# wear_user_id = "9a48a3cb"
# wear_user_id = "nGWEnb4i"
# wear_user_id = "LhDVt7QZ"
wear_user_id = "muQdSf6h"

# set_time = pd.date_range(start='2021-06-20', end='2021-06-22')
set_time = pd.date_range(start='2023-01-29', end='2023-01-30')
# set_time = pd.date_range(start='2020-08-15', end='2020-08-16')
# set_time = pd.date_range(start='2020-11-01', end='2020-11-10')
# set_time = pd.date_range(start='2021-06-01', end='2021-06-12')
set_time = [datetime.strftime(st, '%Y-%m-%d') for st in set_time]

# set_time = ['2020-07-16', '2020-07-17', '2020-07-30', '2020-07-31', '2020-08-11', '2020-09-15', '2020-09-16']
# set_time = ['2020-10-19','2020-10-20']
# set_time = ['2021-01-18','2021-01-19']
# set_time = ['2021-02-05','2021-02-20']
# set_time = ['2021-01-19']

# sql_s_1 = "SELECT WEAR_USER_ID, FROMPPG, DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, tuple(set_time))
# res_1 = ansql.ansql_read_mysql(sql_s_1)
sql_s_2 = "SELECT WEAR_USER_ID, FROMPPG, DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, tuple(set_time))
res_2 = ansql.ansql_read_mysql(sql_s_2)

ppg_history = pd.DataFrame(columns=feature_names_sql + ['wear_user_id', 'ppg_time', 'date'])

for res in [res_2]:
    for a in range(res.shape[0]):
        ppg_list = res.iloc[a]['FROMPPG']

        for ps in ppg_list.split(";"):
            if len(ps) == 0:
                continue
            try:
                # temp_ppg_history = []
                ppg_time, ppg_values = ps.split("/", 1)
                # ppg_values = ps.split("/")[1]
                ppg_values = list(ppg_values.split(","))
                if len(ppg_values) % 12 == 0:
                    # ppg_values = list(ppg_values.split(','))
                    ppg_values = np.array(ppg_values)
                    ppg_values = ppg_values.reshape(-1, 12)
                    temp_ppg_history = pd.DataFrame(data=ppg_values, columns=feature_names_sql)
                    temp_ppg_history = temp_ppg_history.astype('float')

                    temp_ppg_history['wear_user_id'] = wear_user_id
                    temp_ppg_history['ppg_time'] = ppg_time
                    
                    temp_ppg_history['date'] = res.iloc[a]['DATE']
                    # 拼接到 df_results
                    ppg_history = pd.concat([ppg_history, temp_ppg_history], ignore_index=True)
            except Exception as e:
                print(e)

ppg_history = ppg_history.astype({
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
                        'date': 'str'
                        # 'sbp': 'float32',
                        # 'dbp': 'float32'
                        })

# results['Volume'] = results[['AC', 'SystolicTime', 'DiastolicTime']].apply(lambda x: float(x['AC']) * (1.0 + float(x['SystolicTime']) / float(x['DiastolicTime'])))
# 删除为 0 的行
ppg_history.drop(index = ppg_history.Slope[ppg_history.Slope == 0].index, inplace=True)
ppg_history.drop(index = ppg_history.SystolicTime[ppg_history.SystolicTime == 0].index, inplace=True)
ppg_history.drop(index = ppg_history.DiastolicTime[ppg_history.DiastolicTime == 0].index, inplace=True)
ppg_history.reset_index(drop=True, inplace=True)

ppg_history['A1/A2'] = ppg_history['A1'] / ppg_history['A2']
ppg_history['Volume'] = ppg_history.apply(lambda x: float(x['AC']) * (1.0 + float(x['SystolicTime']) / float(x['DiastolicTime'])), axis=1)
ppg_history['Volume/Slope'] = ppg_history['Volume'] / ppg_history['Slope']
ppg_history['Volume/RR'] = ppg_history['Volume'] / ppg_history['RR']
ppg_history['SystolicTime/RR'] = ppg_history['SystolicTime'] / ppg_history['RR']
ppg_history['DiastolicTime/RR'] = ppg_history['DiastolicTime'] / ppg_history['RR']

# print(ppg_history.corr())

ppg_history.to_csv(r"C:\Users\jianbin.xu\Desktop\Andun\用户ppg数据\{}.csv".format(wear_user_id))

################################################
################################################

# 数据统计分析
feature_names = ['Slope','DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime','A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 
                     'A1', 'A2', 'AC']
def analysis(df_data, dates_list):
    # 对每一天的数据进行统计分析
    for dl in dates_list:
        print("*************** {} ****************".format(dl))
        temp_df_data = df_data[df_data['date'] == dl]
        temp_df_data = temp_df_data.astype({'ppg_time': 'float32'})
        # 找出早上6点前的ppg数据
        temp_df_data = temp_df_data[temp_df_data['ppg_time'] <= 60000]
        features_statistics = temp_df_data[feature_names].astype('float32').describe(percentiles=[0.01,0.05,0.1,0.15,0.25,0.33,0.5,0.68,0.75,0.85,0.9,0.95,0.99])
        print(features_statistics)


analysis(ppg_history, set_time)