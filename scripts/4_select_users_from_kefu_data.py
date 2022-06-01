#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
# sys.path.append(r"old_model/")
sys.path.append(r".")
sys.path.append(r"utils/")
sys.path.append(r"personal_models/")
import warnings
warnings.filterwarnings('ignore')

import time
import os
import pandas as pd
import numpy as np
from andun_sql.andun_sql import AndunSql
from config import Min_Minute

from a_1_select_kefu_data_by_user import select_kefu_data_by_user
from a_2_user_data_analysis import pull_bp_features_by_user_from_db, bp_features_analysis
from a_3_find_feature_bp_by_user import find_feature_bp_by_user
from a_4_train_model_lr import model_lr
from a_4_train_model_rf import model_rf
from a_5_predict_all_ppg_data import predict_all_ppg_data


import sys
sys.path.append(r'grpc/proto/')
import numpy as np
import pandas as pd
import grpc
import anbp_pb2, anbp_pb2_grpc

# IP = "192.168.100.20:50056"
# IP = "192.168.100.33:50057"
# IP = "192.168.100.232:50057"
# IP = "39.96.40.181:50057"
# IP = "47.95.198.148:50057"
# IP = "39.97.104.203:50057"

# 线上 NG 的地址
IP = "39.97.198.78:50057"

"""
按照选定的时间点, 筛查客服记录的数据里面, 血压数据比较多的用户,跑一下个人模型试试,看
是否满足要求,如果满足,则加入新模型
"""


ansql = AndunSql()
# c_bp_history 里最早的记录时间
# start_date = "2019-08-24 00:00:00"
# 手动指定时间
# start_date = "2020-09-20 00:00:00"
# start_date = "2020-10-09 00:00:00"
start_date = "2021-03-15 00:00:00"
end_date = "2021-03-16 00:00:00"
# start_date = pd.to_datetime(start_date)

# 找出指定时间段内 记录血压的用户
# sql_select = 'SELECT wear_user_id,gmt_create,create_time FROM andun_cms.c_bp_history WHERE status=0 AND create_time >= "{}" '.format(start_date)
# sql_select = 'SELECT wear_user_id,gmt_create,create_time FROM andun_cms.c_bp_history WHERE status=0 AND create_time BETWEEN "{}" AND "{}" '.format(start_date, end_date)
sql_select = 'SELECT wear_user_id,gmt_create,create_time FROM andun_cms.c_bp_history WHERE create_time BETWEEN "{}" AND "{}" '.format(start_date, end_date)
bp_history_data = ansql.ansql_read_mysql(sql_select)
# 找出用户, 然后找出这些用户的所有血压记录数据
bp_history_users = bp_history_data['wear_user_id'].unique()
if len(bp_history_users) == 0:
    print("没有用户...")
elif len(bp_history_users) == 1:
    sql_select = 'SELECT wear_user_id, gmt_create,create_time FROM andun_cms.c_bp_history WHERE wear_user_id = "{}" '.format(bp_history_users[0])
else:
    sql_select = 'SELECT wear_user_id, gmt_create,create_time FROM andun_cms.c_bp_history WHERE wear_user_id in {} '.format(tuple(bp_history_users))
bp_history_data = ansql.ansql_read_mysql(sql_select)


# 从 d_users_bp_model 里面 找出使用新模型的用户
sql_select_new_model_users = 'SELECT wear_user_id, create_time, update_time FROM andun_watch.d_users_bp_model'
new_model_users = ansql.ansql_read_mysql(sql_select_new_model_users)
new_model_users['wear_user_id'] = new_model_users['wear_user_id'].apply(lambda x: str(x))
new_model_unique_users = new_model_users['wear_user_id'].unique()

# 查询轩轩的模型表
sql_select_xuan_model_users = 'SELECT wear_user_id FROM andun_watch.d_user_bp_nn_model_para'
xuan_model_users = ansql.ansql_read_mysql(sql_select_xuan_model_users)
xuan_model_users['wear_user_id'] = xuan_model_users['wear_user_id'].apply(lambda x: str(x))
xuan_model_unique_users = xuan_model_users['wear_user_id'].unique()



# 找出 用户的 wear_user_id
unique_wear_user_ids = bp_history_data['wear_user_id'].unique()

# 暂时排除的用户
delete_users_path = r"X:\AnDun\work\新血压采集\暂时排除名单.csv"
delete_users = pd.read_csv(delete_users_path, encoding='utf-8')
un_delete_users = delete_users['wear_user_id'].unique()



def train(wear_user_id):
    # 调用线上的接口, 使用新数据重新训练
    user_info = '{"wearUserId": "' + wear_user_id + '","age": 26,"height": 180,"weight": 85,"gender": 1}'
    test_bp_features = ""
    channel = grpc.insecure_channel(IP)
    stub = anbp_pb2_grpc.GreeterStub(channel=channel)
    response = stub.ANBP(anbp_pb2.BPRequest(userInfo = user_info,
                                            oStatus = "-2",
                                            features=test_bp_features,
                                            preReportBP="100,90,90",
                                            #  preReportBP="0,-1",
                                            preReportTimeDiffMinute = "31"))

    print("status: {}, bloodPpressure: {}, timestamp: {}".format(response.status, response.bloodPpressure, response.timestamp))
    return response.status


if len(unique_wear_user_ids) == 0:
    print("没有用户......")
else:
    if len(unique_wear_user_ids) == 1:
        sql_user_info  = "SELECT ID,USERNAME FROM andun_app.a_wear_user WHERE ID = '{}' ".format(unique_wear_user_ids[0])
    else:
        sql_user_info  = "SELECT ID,USERNAME FROM andun_app.a_wear_user WHERE ID in {} ".format(tuple(unique_wear_user_ids))

    user_info = ansql.ansql_read_mysql(sql_user_info)

    print(user_info.head())


    results = []

    # 遍历 bp_history_data 里面的每条数据, 判断是否在新模型表中,如果在的话,判断从加入新模型后,是否有新数据上传,
    for uwu in unique_wear_user_ids:
        temp_result = []

        # 找出此用户的所有记录数据
        this_user_history = bp_history_data[bp_history_data['wear_user_id'] == uwu]

        # 如果此用户在排除名单中, 则略过
        if uwu in un_delete_users:
            print(" {} 此用户在排除名单中...".format(uwu))
            temp_result.append(uwu)
            for a in range(4):
                temp_result.append(None)
            temp_result.append(len(this_user_history))
            temp_result.append("此用户在排除名单中...")
            results.append(temp_result)
            continue
        elif uwu in xuan_model_unique_users:
            print(" {} 此用户在轩轩的模型中...".format(uwu))
            temp_result.append(uwu)
            for a in range(4):
                temp_result.append(None)
            temp_result.append(len(this_user_history))
            temp_result.append("此用户在轩轩的模型中...")
            results.append(temp_result)
            # 从模型中删除
            ansql.delete_d_user_bp_nn_model_para(unique_wear_user_ids[0])
            continue

        
        if uwu in new_model_unique_users:
            # 找出此用户被加入新模型的时间
            # create_time = new_model_users[new_model_users['wear_user_id'] == uwu]['create_time'].tolist()[-1]
            update_time = new_model_users[new_model_users['wear_user_id'] == uwu]['update_time'].tolist()[-1]
            # 判断加入新模型后,此用户有没有上传血压数据
            this_user_history = this_user_history[this_user_history['gmt_create'] > update_time]
            # print(" {} 此用户已加入新血压模型, 加入新血压模型后, 上传了 {} 组血压数据...".format(uwu, len(this_user_history)))
            temp_result.append(uwu)
            for a in range(3):
                temp_result.append(None)
            temp_result.append("新模型用户")
            temp_result.append(len(this_user_history))

            if len(this_user_history) > 0:
                status = train(uwu)
                if str(status) == '2':
                    # 修改 d_users_bp_model 的 update_time值
                    new_update_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    ansql.update_d_users_bp_model(uwu, new_update_time)
                    temp_result.append("已重新训练...")
                else:
                    temp_result.append("重新训练 失败...")
                    # 删除
                    ansql.delete_d_users_bp_model(unique_wear_user_ids[0])
            else:
                temp_result.append("不需要重新训练 ...")

            results.append(temp_result)

        else:
            # 此用户不在新模型里面, 统计此用户记录数据的数量
            this_user_history.sort_values(by=['gmt_create'], ascending=True, inplace=True)
            # 找出记录数据的 最早的时间
            start_time = this_user_history['gmt_create'].tolist()[0]
            # 找出记录数据的 最晚的时间
            end_time = this_user_history['gmt_create'].tolist()[-1]

            temp_result.append(uwu)
            temp_result.append(len(this_user_history))
            temp_result.append(start_time)
            temp_result.append(end_time)

            if len(this_user_history) > 9:
                print(" {} 此用户不在新血压模型中, 共记录了 {} 条数据, 最早时间: {}, 最晚时间: {}".format(uwu, len(this_user_history), start_time, end_time))
                ############################
                ############################
                # 尝试对此用户进行个人模型建模
                # 查询此用户的客服记录的血压值记录
                select_kefu_data_by_user(wear_user_id=uwu)
                # 根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据
                try:
                    un_sbp, un_dbp = find_feature_bp_by_user(wear_user_id=uwu, min_minute=Min_Minute)
                except Exception as e:
                    print(e)
                    print("匹配客服记录的数据发生错误...")

                try:
                    if len(un_sbp) > 10:
                        print(" {} 此用户可用的血压数据数量...{}".format(uwu, len(un_sbp)))
                        # 查询此用户的 所有的血压 ppg_feature 记录值,用于统计分析
                        # pull_bp_features_by_user_from_db(wear_user_id=uwu, last_n_day=10)
                        # # 统计分析此用户的 ppg_feature值 的统计分析, 用于后面的过滤数据使用
                        # bp_features_analysis(wear_user_id=uwu)
                        # # 训练 personal_model
                        # # model_lr(wear_user_id=uwu)
                        # model_rf(wear_user_id=uwu)
                        status = train(uwu)
                        if str(status) == '2':
                            new_update_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                            ansql.insert_d_users_bp_model(uwu, new_update_time)
                            print(" {} 已加入新模型...".format(uwu))
                        else:
                            print(" {} 加入新模型失败...".format(uwu))

                        temp_result.append("数据个数-{}, 训练结果-{}".format(len(un_sbp), status))
                    else:
                        temp_result.append("数据个数-{}".format(len(un_sbp)))
                    ############################
                    ############################
                    # temp_result.append("数据个数-{}, 训练结果-{}".format(len(un_sbp), status))
                    temp_result.append(un_sbp)
                    temp_result.append(un_dbp)
                except Exception as e:
                    print(e)
                    temp_result.append("数据处理及模型训练错误!")
                    temp_result.append(None)
                    temp_result.append(None)
            
            else:
                temp_result.append(None)
                temp_result.append(None)
                temp_result.append(None)

            results.append(temp_result)

    results = pd.DataFrame(data=results, columns=['wear_user_id', 'num', 'start_time', 'end_time','num_in_use','sbp_values', 'dbp_values'])
    results.sort_values(by=['num'], ascending=False, inplace=True)
    print(results)
    results.to_csv(r"scripts/4_results_2021-03-26_01.csv", encoding='utf-8', index=False)

    # count_results = bp_history_data['wear_user_id'].value_counts()
    # results = pd.merge(count_results, user_info, left_on='Unnamed: 0', right_on='ID', how='left')
    # # RES = count_results.merge(name,left_on='wear_user_id',right_on='ID',how='inner')
    # # RES.to_csv("./RES.csv", encoding='utf-8')
    # print(results.head())
