#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(r".")
sys.path.append(r"utils/")
sys.path.append(r'grpc/proto/')
import numpy as np
import pandas as pd
np.random.seed(666)
import copy
from datetime import datetime

from andun_sql.andun_sql import AndunSql
from sklearn.metrics import mean_squared_error # MSE , RMSE对MSE开平方即可
from sklearn.metrics import mean_absolute_error # MAE
import joblib
import json
import pickle
from utils import cal_acc, add_new_features, remove_max_min, clean_data_with_quantile_statistics, clean_data_with_mean_std
from utils import plot_sbp_dbp_by_ppg_time
# from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from tqdm import tqdm
# from models.predictor import Predictor
import grpc
import anbp_pb2, anbp_pb2_grpc


"""
使用线上的grpc服务,进行血压预测
对此用户的所有ppg数据进行预测, 每个时间点的 ppg 信号预测 的结果取平均, 作为此时刻的血压值
按照给定的 date range 进行预测,不是预测此用户的所有历史 ppg feature 数据
"""

# 需要配置的参数
wear_user_id = "902170160200952"
start_date = '2022-05-30'
end_date = '2022-06-01'


# IP = "192.168.100.24:50056"
# IP = "47.94.169.147:50056"
# IP = "39.107.44.4:50056"
# IP = "39.107.40.248:50056"

# IP = "192.168.100.122:50056"
# IP = "192.168.100.232:50057"
# IP = "39.96.40.181:50057"
# IP = "47.95.198.148:50057"
# IP = "39.97.104.203:50057"

# 线上 NG 的地址 个人模型
# IP = "39.97.198.78:50057"

# 武轩 血压模型 的地址
# IP = "172.17.224.119:50060"

# 旧模型 NG 的地址
IP = "47.95.230.250:50056"


# predictor = Predictor(wear_user_id=wear_user_id)

# train_data_path = "personal_models/data/{}_ppg_data.csv".format(wear_user_id)
# ppg_values_for_test = "personal_models/data/{}_ppg_history_split.csv".format(wear_user_id)
# 把此用户所有的ppg信号数据进行 预测
# data_for_test = pd.read_csv(ppg_values_for_test)
# feature_statistics_path = "personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id)
# feature_statistics = pd.read_csv(feature_statistics_path, header=0, index_col=0)
# feature_names_new = FEATURE_NAME_NEW

ansql = AndunSql()
# 查询 此用户的基本信息
user_info = ansql.ansql_user_info(wear_user_id)
age = user_info['Age'].tolist()[0]
gender = user_info['Gender'].tolist()[0]
height = user_info['Height'].tolist()[0]
weight = user_info['Weight'].tolist()[0]


user_info = {
    "wearUserId": wear_user_id,
    "age": age,
    "height": height,
    "weight": weight,
    "gender": gender
}
user_info = json.dumps(user_info)

"""
对此用户的所有ppg数据进行预测, 每条ppg 信号预测一个值
"""

# 保存 每一组 ppg_feature 的预测结果
df_test = []

set_date_range = pd.date_range(start=start_date, end=end_date)
set_date_range = [datetime.strftime(st, '%Y-%m-%d') for st in set_date_range]


# count = 0
last_sbp = None
last_dbp = None
last_cli = None
last_time = 0
time_differ_minute = None
# 遍历 每个 ppg_value数据
for sdr in tqdm(set_date_range):
    print("date: {}".format(sdr))
    # 一 从表里面拿当天的ppg
    # temp_dft = data_for_test[data_for_test['date'] == sdr]
    # 二 从数据库里面取
    temp_dft = ansql.ansql_bp_feature(wear_user_id, [sdr])

    # 判断当天是否有数据
    if temp_dft.shape[0] == 0:
        continue
    
    # 遍历当天的每条 ppg_value 数据
    for a in range(temp_dft.shape[0]):
        # 第一次调用的时候
        if a == 0:
            last_sbp = 130
            last_dbp = 80
            last_cli = 100
            time_differ_minute = 32
        else:
            # 计算 time_differ_minute
            time_differ_minute = int(temp_dft.iloc[a]['ppg_time']) - int(last_time)
            time_differ_minute = int(abs(time_differ_minute / 60))

        ppg_values_all = temp_dft.iloc[a]['FROMPPG']

        for ps in ppg_values_all.split(";"):
            if len(ps) == 0:
                    continue
            ppg_time, ppg_values = ps.split("/", 1)
            # 把ppg_values保存成 dataframe
            temp_result = []
            
            # 添加个人的信息特征
            temp_result.append(age)
            temp_result.append(height)
            temp_result.append(weight)
            temp_result.append(gender)

            ##################################
            ##################################
            # 调用server
            channel = grpc.insecure_channel(IP)
            stub = anbp_pb2_grpc.GreeterStub(channel=channel)
            response = stub.ANBP(anbp_pb2.BPRequest(userInfo = user_info,
                                                    oStatus = "4",
                                                    features=ppg_values,
                                                    preReportBP="{},{},{}".format(last_sbp, last_dbp, last_cli),
                                                    #  preReportBP="0,-1",
                                                    preReportTimeDiffMinute = "{}".format(time_differ_minute)))

            print("status: {}, bloodPpressure: {}, timestamp: {}".format(response.status, response.bloodPpressure, sdr + " " + ppg_time[0:2] + ":" + ppg_time[2:4]))
            # print("status: {}, bloodPpressure: {}, timestamp: {}".format(response.status, response.bloodPpressure, response.timestamp))
            pred_result = response.bloodPpressure

            if pred_result:
                pred_result = pred_result.split("/")
                temp_result.append(pred_result[0])
                temp_result.append(pred_result[1])

                # 更新 上一次的预测数据
                last_sbp = pred_result[0]
                last_dbp = pred_result[1]
                last_cli = pred_result[2]
                last_time = ppg_time
            else:
                temp_result.append(None)
                temp_result.append(None)

            # 加入其它信息字段
            temp_result.append(wear_user_id)
            temp_result.append(temp_dft.iloc[a]['DATE'])
            temp_result.append(ppg_time)
            # 拼接到 df_results
            # df_test = pd.concat([df_test, temp_result], ignore_index=True)
            df_test.append(temp_result)

# df_test.to_csv("predict_results/{}_pred_weights01.csv".format(wear_user_id), index=False)
# # 每个时间点有多组的 ppg_feature, 只保留一组, 便于查看每个时间点的模型预测值
# df_test.drop_duplicates(subset=['weight_sbp', 'weight_dbp'], keep='first', ignore_index=True,  inplace=True)
df_test = pd.DataFrame(data=df_test, columns=['Age', 'Height', 'Weight', 'Gender','pred_sbp', 'pred_dbp', 'wear_user_id', 'date', 'ppg_time'])
df_test.to_csv("predict_results/{}_pred_weights02_date_range_online.csv".format(wear_user_id), index=False)
