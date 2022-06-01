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
从数据库中 找出客服记录的用户血压数据,  按照设置的wear_user_id 查找, 
并找出每一条数据 当天对应的 ppg feature, 保存
"""



def select_kefu_data_by_user(wear_user_id):

    ansql = AndunSql()

    # 按照设置的时间查出 所有优化过的用户
    # sql_select_users = 'SELECT wear_user_id, gmt_create, sbp, dbp FROM andun_cms.c_bp_history WHERE wear_user_id="{}" AND create_time >= "{}";'.format(wear_user_id, set_time)
    # sql_select_users = 'SELECT wear_user_id, gmt_create, sbp, dbp FROM andun_cms.c_bp_history WHERE wear_user_id="{}" AND enabled=1;'.format(wear_user_id)
    sql_select_users = 'SELECT wear_user_id, gmt_create, sbp, dbp FROM andun_cms.c_bp_history WHERE wear_user_id="{}";'.format(wear_user_id)
    select_users = ansql.ansql_read_mysql(sql_select_users)
    select_users['wear_user_id'].astype(str)
    select_users.to_csv("personal_models/data/{}_kefu_history.csv".format(wear_user_id), index=False)

    print("select_users.shape:",select_users.shape)
    print(select_users.dtypes)

    # 把查询的结果, 遍历每个用户,每一天,查找出此用户当天的血压ppg信号的12个特征记录
    select_users['day'] = select_users['gmt_create'].apply(lambda x: str(x)[0:10])
    # 对day取unique()
    unique_days = select_users['day'].unique()


    ppg_history = []

    # 遍历 unique_days
    for ud in unique_days:
        print("{} - {}".format(wear_user_id, ud))        
        # temp_ud = uu_bp_feature[uu_bp_feature['DATE'] == datetime.date(datetime.strptime(str(ud), '%Y-%m-%d'))]
        temp_ud = ansql.ansql_bp_feature_train(wear_user_id, [ud])

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

        # break
        time.sleep(0.6)


    # 把 ppg_history整理成 dataframe
    df_ppg_history = pd.DataFrame(data=ppg_history, columns=['wear_user_id', 'ppg_time', 'ppg_values', 'date'])
    # wear_user_id 置为  str  5915e225
    df_ppg_history['wear_user_id'].astype(str)
    df_ppg_history.to_csv("personal_models/data/{}_ppg_history.csv".format(wear_user_id), index=False)
    print(df_ppg_history.shape)
    print(df_ppg_history.head())


if __name__ == "__main__":

    wear_user_id = "5915e225"
    # set_time = "2020-05-01 06:00:00"
    select_kefu_data_by_user(wear_user_id=wear_user_id)