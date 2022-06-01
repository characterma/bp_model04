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

"""
从数据库中 找出客服记录的用户血压数据,并找出每一条数据对应的feature,用此数据去评估模型
"""

# 根据设置的时间, 查找客服记录的 用户血压数据记录
start_time = "2020-01-01 00:00:00"
end_time = "2020-08-12 23:59:59"
ansql = AndunSql()

select_users_path = r"data/raw_data/data_kefu/select_users.csv"

if not os.path.exists(select_users_path):
    # 按照设置的时间查出 所有优化过的用户
    sql_select_users = 'SELECT wear_user_id, gmt_create, sbp, dbp, create_time FROM andun_cms.c_bp_history WHERE create_time >= "{}" and create_time <= "{}";'.format(start_time, end_time)
    select_users = ansql.ansql_read_mysql(sql_select_users)
    select_users.to_csv(select_users_path, index=False)
else:
    select_users = pd.read_csv(select_users_path)

print("select_users.shape:",select_users.shape)
print(select_users.dtypes)




# 把查询的结果, 遍历每个用户,每一天,查找出此用户当天的血压ppg信号的12个特征记录
select_users['day'] = select_users['gmt_create'].apply(lambda x: str(x)[0:10])
select_users_groupby = select_users[['wear_user_id', 'day']].copy(deep=True)

unique_users = select_users_groupby['wear_user_id'].unique()

ppg_history = []

for uu in unique_users:
    # 找出此用户的血压记录的天
    this_user_bp_history = select_users[select_users['wear_user_id'] == uu]
    # 对day取unique()
    unique_days = this_user_bp_history['day'].unique()
    # 从数据库中取出此用户,这些天里面的ppg信号值记录
    uu_bp_feature = ansql.ansql_bp_feature(uu, tuple(unique_days))
    # print(uu_bp_feature['DATE'].values)
    if not isinstance(uu_bp_feature, pd.DataFrame):
        print("{} 的ppg数据没有... {} ".format(uu, unique_days))
        time.sleep(1.0)
        continue

    # 遍历 unique_days
    for ud in unique_days:
        print("{} - {}".format(uu, ud))        
        temp_ud = uu_bp_feature[uu_bp_feature['DATE'] == datetime.date(datetime.strptime(str(ud), '%Y-%m-%d'))]
        if temp_ud.shape[0] > 0:
        
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
                        temp_ppg_history.append(uu)
                        temp_ppg_history.append(ppg_time)
                        temp_ppg_history.append(",".join(ppg_values))
                        temp_ppg_history.append(ud)

                        ppg_history.append(temp_ppg_history)
                    else:
                        print("此用户{}当天{}ppg数据长度有误...".format(uu, ud))
                
                except Exception as e:
                    print(e)
                    continue
        else:
            print("此用户{}当天{}无ppg数据...".format(uu, ud))

    time.sleep(1.0)


# 把 ppg_history整理成 dataframe
df_ppg_history = pd.DataFrame(data=ppg_history, columns=['wear_user_id', 'ppg_time', 'ppg_values', 'date'])
df_ppg_history.to_csv(r"data/raw_data/data_kefu/kefu_ppg_history.csv", encoding='utf-8', index=False)
print(df_ppg_history.shape)
print(df_ppg_history.head())


# select_users_groupby.drop_duplicates(inplace=True, ignore_index=True)
# select_users_groupby.sort_values(by='wear_user_id', inplace=True)
# print(select_users_groupby.head())
# print(select_users_groupby.shape)
