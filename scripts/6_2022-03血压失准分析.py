#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import groupby
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(r".")
sys.path.append(r"utils/")
sys.path.append(r"personal_models/")
import numpy as np
import pandas as pd
from andun_sql.andun_sql import AndunSql
import time


# wrong_bp_users = pd.read_excel(r"2022-03血压失准反馈.xlsx")
wrong_bp_users = pd.read_csv(r"血压失准分析/2022-04血压失准反馈.csv", dtype={'A_DEVICE_ID':'str'})
wrong_bp_users.fillna('0', inplace=True)
wrong_bp_users['A_DEVICE_ID'] = wrong_bp_users['A_DEVICE_ID'].apply(lambda x: x.strip())
wrong_bp_users = wrong_bp_users[['A_DEVICE_ID', '产品代别型号']]
# wrong_bp_users.drop_duplicates(subset=['设备编号'], inplace=True,keep='first')
print(wrong_bp_users.shape)
print(wrong_bp_users.head())

wrong_bp_users['count'] = 1
groupby_data = wrong_bp_users[['A_DEVICE_ID', 'count']].groupby('A_DEVICE_ID', as_index=False).agg({'count':'count'})

wrong_bp_users.drop(axis=1, columns=['count'], inplace=True)
wrong_bp_users.drop_duplicates(subset=['A_DEVICE_ID'], inplace=True,keep='first')

wrong_bp_users = pd.merge(wrong_bp_users, groupby_data, how='inner', on='A_DEVICE_ID')
print(wrong_bp_users.shape)
print(wrong_bp_users.head())

# 按照设备号查找 wear_user_id
ansql = AndunSql()
devices = wrong_bp_users['A_DEVICE_ID'].to_list()
select_sql = "SELECT A_WEAR_USER_ID, A_DEVICE_ID, CREATE_TIME FROM andun_app.t_wear_med_device WHERE A_DEVICE_ID in {}".format(tuple(devices))
users_device = ansql.ansql_read_mysql(select_sql)
users_device.to_csv("血压失准分析/devices.csv", index=False)




# users_device = pd.read_csv("血压失准分析/devices.csv", dtype={'A_DEVICE_ID':'str'})
# users_device.sort_values(by=['CREATE_TIME'], inplace=True, ascending=False)
# users_device.drop_duplicates(subset=['A_DEVICE_ID'], inplace=True, keep='first')
print(users_device.shape)
print(users_device.head())


# wrong_bp_users = pd.merge(wrong_bp_users, users_device, how='outer', left_on='设备编号', right_on='A_DEVICE_ID')
wrong_bp_users = pd.merge(wrong_bp_users, users_device, how='outer',on='A_DEVICE_ID')
# wrong_bp_users = wrong_bp_users.merge(users_device, how='outer', on='A_DEVICE_ID')
print(wrong_bp_users.shape)
print(wrong_bp_users.head())

wrong_bp_users['bp'] = None
wrong_bp_users['总数'] = 0
wrong_bp_users['有效血压'] = 0
wrong_bp_users['无效血压'] = 0
for a in range(wrong_bp_users.shape[0]):
    wear_user_id = wrong_bp_users.iloc[a]['A_WEAR_USER_ID']

    print(a, wear_user_id)
    if wear_user_id:
        select_sql = "SELECT * FROM andun_cms.c_bp_history WHERE wear_user_id = '{}'".format(wear_user_id)
        bp_history = ansql.ansql_read_mysql(select_sql)
        bp_history['enabled'] = bp_history['enabled'].apply(lambda x: int().from_bytes(x, byteorder='big', signed=True))
        print(bp_history.head())

        wrong_bp_users.loc[a,'bp'] = '[{}, {}]'.format(bp_history['sbp'].to_list(), bp_history['dbp'].to_list())
        wrong_bp_users.loc[a, '总数'] = bp_history.shape[0]
        wrong_bp_users.loc[a, '有效血压'] = len(bp_history[bp_history['enabled'] == 0])
        wrong_bp_users.loc[a, '无效血压'] = len(bp_history[bp_history['enabled'] == 1])

    # break
    time.sleep(0.5)

wrong_bp_users['有效数据占比'] = wrong_bp_users['有效血压'] / wrong_bp_users['总数']
wrong_bp_users.to_csv("血压失准分析/wrong_bp_users_04.csv", index=False)