#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append(r'utils/')
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from config import SIGMA, Min_Quantile, Max_Quantile
# from config import diff_limit_sbp, diff_limit_dbp
from config import time_interval, limit_percentage
import requests
import json
from dingtalkchatbot.chatbot import DingtalkChatbot
from andun_sql.andun_sql import AndunSql


# 统计预测值和真实值之间的偏差, <3, <5, <7, <9的占比
def cal_acc(y_real, y_pred):

    df = pd.DataFrame()
    df['y_real'] = y_real
    df['y_pred'] = y_pred

    # df['diff'] = df[['y_real', 'y_pred']].apply(lambda x: abs(int(x[0]) - int(x[1])))
    df['diff'] = abs(df['y_real'] - df['y_pred'])

    for diff in [3, 5, 7, 11, 13, 15]:
        temp_diff = df[df['diff'] < diff]
        print("偏差小于 {} 的占比为 {}".format(diff, temp_diff.shape[0] / df.shape[0]))



def cal_time_delta(x, refer_time):

    time_delta = x - refer_time
    if time_delta.days < 0:
        time_delta = refer_time - x
    return time_delta



# 此函数用于扩充指定范围内的 血压数据
def enlarge(train_data, sbp_zone, sbp_scope, dbp_scope):
    data_zone = train_data[train_data['sbp_zone']==sbp_zone]
    df_enlarge = pd.DataFrame(columns=train_data.columns)
    for a in range(data_zone.shape[0]):
        print("a: {}".format(a))
        temp = data_zone.iloc[a]

        sbp = int(data_zone.iloc[a]['AvgSBP'])
        dbp = int(data_zone.iloc[a]['AvgDBP'])

        for s in range(sbp-sbp_scope, sbp+sbp_scope+1):
            for d in range(dbp-dbp_scope, dbp+dbp_scope):

                temp['AvgSBP'] = s
                temp['AvgDBP'] = d
                df_enlarge = df_enlarge.append(temp, ignore_index=True)
    return df_enlarge



# 此函数用于扩充指定范围内的  DBP 只单边扩展
def enlarge_double(train_data, sbp_zone, sbp_scope, dbp_scope):
    data_zone = train_data[train_data['sbp_zone']==sbp_zone]
    df_enlarge = pd.DataFrame(columns=train_data.columns)
    for a in range(data_zone.shape[0]):
        print("a: {}".format(a))
        temp = data_zone.iloc[a]

        sbp = int(data_zone.iloc[a]['AvgSBP'])
        dbp = int(data_zone.iloc[a]['AvgDBP'])

        for s in range(sbp-sbp_scope, sbp+sbp_scope+1):
            for d in range(dbp-dbp_scope, dbp):

                temp['AvgSBP'] = s
                temp['AvgDBP'] = d
                df_enlarge = df_enlarge.append(temp, ignore_index=True)
    return df_enlarge



# # 在原来 ppg 特征的基础上,构建新特征
# def add_new_features(train_data):
#     # ['A1-A2', 'DT/ST', 'DT-ST', 'A1/RR', 'A2/RR', 'UT/ST','UT/RR', 'ST/RR', 'DT/RR', 'ST*AC', 'A1/(ST*AC)', 'DT*AC', 'A2/(DT*AC)']
#     train_data['A1-A2'] = train_data['A1'] - train_data['A2']
#     train_data['DT/ST'] = train_data['DiastolicTime'] / train_data['SystolicTime']
#     train_data['DT-ST'] = train_data['DiastolicTime'] - train_data['SystolicTime']
#     train_data['A1/RR'] = train_data['A1'] / train_data['RR']
#     train_data['A2/RR'] = train_data['A2'] / train_data['RR']
#     train_data['UT/ST'] = train_data['UpToMaxSlopeTime'] / train_data['SystolicTime']
#     train_data['UT/RR'] = train_data['UpToMaxSlopeTime'] / train_data['RR']
#     train_data['ST/RR'] = train_data['SystolicTime'] / train_data['RR']
#     train_data['DT/RR'] = train_data['DiastolicTime'] / train_data['RR']
#     train_data['ST*AC'] = train_data['SystolicTime'] * train_data['AC']
#     train_data['A1/(ST*AC)'] = train_data['A1'] / train_data['ST*AC']
#     train_data['DT*AC'] = train_data['DiastolicTime'] * train_data['AC']
#     train_data['A2/(DT*AC)'] = train_data['A2'] / train_data['DT*AC']

#     return train_data


# 在原来 ppg 特征的基础上,构建新特征, 只使用比值特征
def add_new_features(train_data):
    # ['(A1-A2)/(A1+A2)', 'DT/ST', 'A1/RR', 'A2/RR', 'UT/ST','UT/RR', 'ST/RR', 'DT/RR', 'A1/(ST*AC)', 'A2/(DT*AC)']

    train_data['A1/AC'] = train_data['A1'] / train_data['AC']
    train_data['A2/AC'] = train_data['A2'] / train_data['AC']
    train_data['A1+A2'] = train_data['A1'] + train_data['A2']
    train_data['A1/(A1+A2)'] = train_data['A1'] / train_data['A1+A2']
    train_data['A2/(A1+A2)'] = train_data['A2'] / train_data['A1+A2']

    train_data['(A1-A2)/(A1+A2)'] = (train_data['A1'] - train_data['A2']) / (train_data['A1'] + train_data['A2'])
    train_data['DT/ST'] = train_data['DiastolicTime'] / train_data['SystolicTime']

    # train_data['A1/RR'] = train_data['A1'] / train_data['RR']
    # train_data['A2/RR'] = train_data['A2'] / train_data['RR']

    train_data['UT/ST'] = train_data['UpToMaxSlopeTime'] / train_data['SystolicTime']
    train_data['UT/RR'] = train_data['UpToMaxSlopeTime'] / train_data['RR']
    train_data['ST/RR'] = train_data['SystolicTime'] / train_data['RR']
    train_data['DT/RR'] = train_data['DiastolicTime'] / train_data['RR']
    train_data['ST*AC'] = train_data['SystolicTime'] * train_data['AC']
    train_data['A1/(ST*AC)'] = train_data['A1'] / train_data['ST*AC']
    train_data['DT*AC'] = train_data['DiastolicTime'] * train_data['AC']
    train_data['A2/(DT*AC)'] = train_data['A2'] / train_data['DT*AC']
    
    # 2020-09-27 新加
    # ppg_history['A1/A2'] = ppg_history['A1'] / ppg_history['A2']
    train_data['Volume'] = train_data.apply(lambda x: float(x['AC']) * (1.0 + float(x['SystolicTime']) / float(x['DiastolicTime'])), axis=1)
    train_data['Volume/Slope'] = train_data['Volume'] / train_data['Slope']
    train_data['Volume/RR'] = train_data['Volume'] / train_data['RR']

    train_data.drop(columns=['ST*AC', 'DT*AC', 'A1+A2'], inplace=True)

    return train_data



def remove_max_min(pred):

    # 对预测的结果进行处理,去除最大值最小值
    # 首先把pred里面 <0 的值 置为0
    # 如果len(pred) <3的话,直接返回,不做处理
    # pred = [p if p>=0 else 0 for p in pred]
    pred = list(pred)
    if len(pred) < 3:
        return pred
    else:
        pred.pop(pred.index(min(pred)))
        pred.pop(pred.index(max(pred)))
    
    return pred



def clean_data_with_quantile_statistics(ppg_features, features_statistics):

    for fsc in features_statistics.columns:
    # if fsc not in ['id', 'Gender_0','Gender_1', 'Age', 'Height', 'Weight', 'AvgSBP','AvgDBP']:
        # if fsc not in ['Unnamed: 0']:
        # print("feature_name:", fsc)
        # 删除 小于 Min_Quantile 的值
        ppg_features = ppg_features[ppg_features[fsc] >= features_statistics.loc[Min_Quantile, fsc]]
        # 删除 大于 Max_Quantile 的值
        ppg_features = ppg_features[ppg_features[fsc] <= features_statistics.loc[Max_Quantile, fsc]]

    return ppg_features



def clean_data_with_mean_std(ppg_features, features_statistics):

    for fsc in features_statistics.columns:
        mean_value = features_statistics.loc['mean', fsc]
        std_value = features_statistics.loc['std', fsc]

        ppg_features = ppg_features[ppg_features[fsc] >= (mean_value - SIGMA*std_value)]
        # 删除 大于 85%的值
        ppg_features = ppg_features[ppg_features[fsc] <= (mean_value + SIGMA*std_value)]

    return ppg_features


def plot_sbp_dbp_by_ppg_time(df_ppg_time_sbp_dbp):

    # ppg_time 转化为 str
    df_ppg_time_sbp_dbp['strif_time'] = df_ppg_time_sbp_dbp['ppg_time'].apply(lambda x: str(x).rjust(6,'0'))
    df_ppg_time_sbp_dbp['strif_time'] = df_ppg_time_sbp_dbp['strif_time'].apply(lambda x: str_insert(x, 2, ':'))
    df_ppg_time_sbp_dbp['strif_time'] = df_ppg_time_sbp_dbp['strif_time'].apply(lambda x: str_insert(x, 5, ':'))
    # "%Y-%m-%d %H:%M:%S"
    # df_ppg_time_sbp_dbp['strif_time'] = df_ppg_time_sbp_dbp['strif_time'].apply(lambda x: datetime.strftime(x, '%H:%M:%S'))
    df_ppg_time_sbp_dbp['strif_time'] = df_ppg_time_sbp_dbp['strif_time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))

    print(df_ppg_time_sbp_dbp.head())
    plt.figure(figsize=(18, 3))
    plt.plot(df_ppg_time_sbp_dbp['strif_time'], df_ppg_time_sbp_dbp['weight_sbp'], 'bo-', markevery=100)
    plt.xlabel('time')
    plt.ylabel('weight_sbp')
    plt.show()

def str_insert(my_str, position, insert_str):

    my_str_list = list(my_str)
    my_str_list.insert(position, insert_str)
    return "".join(my_str_list)



# 计算PPG特征置信度, 可以确定当前佩戴质量
def check_ppg_feature_vaild(ppg_feature):
    # 上传的feature长度小于0, 置信度为0
    if 0 >= len(ppg_feature):
        return 0, None

    ppg_feature = ppg_feature.replace(';', ',')
    feature = [float(i) for i in ppg_feature.split(',')]
    # 上传的feature数组长度错误, 置信度为0
    if 0 != len(feature) % 12:
        return 0, None

    feature = np.array(feature).reshape(len(feature)//12, 12)
    # 上传的feature数组小于3组, 置信度为10
    if 3 > feature.shape[0]:
        return 10, feature

    hr_mean = int(60 / np.mean(sorted(feature[:, 3])[1:-1]))
    # feature数组大于心率, 置信度为100
    if hr_mean <= feature.shape[0]*8:
        return 100, feature

    if 20 >= (hr_mean - feature.shape[0]*8):
        return 100 - (hr_mean - feature.shape[0]*8), feature
    elif 40 >= (hr_mean - feature.shape[0]*8):
        return int(100 - ((5/4) * (hr_mean - feature.shape[0]*8))), feature
    else:
        res = int(100 - (2 * (hr_mean - feature.shape[0]*8)))
        if res <= 0:
            return 0, feature
        else:
            return res, feature

# 根据上一次预测的时间和血压值,调整本次预测的输出值
def adjust_bp(sbp, dbp, pre_sbp, pre_dbp, preReportTimeDiffMinute):

    preReportTimeDiffMinute = int(preReportTimeDiffMinute)

    # 根据 preReportTimeDiffMinute 调整血压
    for ti, lp in zip(time_interval, limit_percentage):
        if preReportTimeDiffMinute <= ti:
            diff_range_sbp = (sbp - pre_sbp) / pre_sbp
            diff_range_dbp = (dbp - pre_dbp) / pre_dbp
            # 判断 sbp 的变化范围
            if abs(diff_range_sbp) >= lp:
                sbp = pre_sbp + (diff_range_sbp/abs(diff_range_sbp)) * pre_sbp * lp
            # 判断 dbp 的变化范围
            if abs(diff_range_dbp) >= lp:
                dbp = pre_dbp + (diff_range_dbp/abs(diff_range_dbp)) * pre_dbp * lp

            break
        
    sbp = round(sbp)
    dbp = round(dbp)
    
    # 增加一个判断 如果低压与高压 压差小于30,则低压使用高压修正
    if sbp - dbp < 30:
        dbp = sbp - 30

    return sbp, dbp

# 判断用户是否在武轩模型
def is_in_wx_model(wear_user_id):
    ansql = AndunSql()
    sql_s = "SELECT * FROM andun_watch.d_user_bp_nn_model_para WHERE wear_user_id='{}'".format(wear_user_id)
    res = ansql.ansql_read_mysql(sql_s)
    if len(res) > 0:
        return 1
    else:
        return 0


# 发送钉钉预警
def send_ding_message(message):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=1ada645a8c260a1f4942428ae4915a4c2006b6b0b831de182c4bcd62df4ffc81"
    xiaoding = DingtalkChatbot(webhook)
    message = "安顿算法部ERROR报警群" + ": " + message
    results = xiaoding.send_text(msg=message, is_at_all=False)
    print(results)



if __name__ == "__main__":
    send_ding_message("血压模型测试...请忽略")