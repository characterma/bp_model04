#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(r".")
sys.path.append(r"utils/")
import numpy as np
import pandas as pd
np.random.seed(666)
import copy

from andun_sql.andun_sql import AndunSql
from sklearn.metrics import mean_squared_error # MSE , RMSE对MSE开平方即可
from sklearn.metrics import mean_absolute_error # MAE
import joblib
import pickle
from utils import cal_acc, add_new_features, remove_max_min, clean_data_with_quantile_statistics, clean_data_with_mean_std
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from tqdm import tqdm
from models.predictor import Predictor


"""
对此用户的所有ppg数据进行预测, 每个时间点的 ppg 信号预测 的结果取平均, 作为此时刻的血压值
"""

wear_user_id = "33f43701"
predictor = Predictor(wear_user_id=wear_user_id)

ppg_values_for_test = "personal_models/data/{}_ppg_history_split.csv".format(wear_user_id)
feature_statistics_path = "personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id)
feature_statistics = pd.read_csv(feature_statistics_path, header=0, index_col=0)
feature_names_new = FEATURE_NAME_NEW

ansql = AndunSql()
# 查询 此用户的基本信息
user_info = ansql.ansql_user_info(wear_user_id)
age = user_info['Age'].tolist()[0]
gender = user_info['Gender'].tolist()[0]
height = user_info['Height'].tolist()[0]
weight = user_info['Weight'].tolist()[0]

"""
对此用户的所有ppg数据进行预测, 每条ppg 信号预测一个值
"""
# 把此用户所有的ppg信号数据进行 预测
data_for_test = pd.read_csv(ppg_values_for_test)

# 保存 每一组 ppg_feature 的预测结果
df_test = pd.DataFrame(columns=feature_names + ['wear_user_id', 'date', 'ppg_time'])

# 保存每条记录的预测结果
# df_results = pd.DataFrame(columns=['wear_user_id', 'date', 'ppg_time'])

# 遍历 每个 ppg_value数据
for a in tqdm(range(data_for_test.shape[0])):
# for a in range(data_for_test.shape[0]):

    print(data_for_test.iloc[a]['date'])

    ppg_values = data_for_test.iloc[a]['ppg_values']
    ppg_values = list(ppg_values.split(','))
    ppg_values = np.array(ppg_values)
    ppg_values = ppg_values.reshape(-1, 12)

    # 把ppg_values保存成 dataframe
    temp_result = pd.DataFrame(data=ppg_values, columns=feature_names_sql)
    temp_result = temp_result.astype('float')

    # 把 UpToMaxSlopeTime 为 0 的值去掉
    # print(temp_result.shape)
    # temp_result = temp_result[temp_result['UpToMaxSlopeTime'] > 0]
    # print(temp_result.shape)

    # 增加新特征
    temp_result = add_new_features(temp_result)
    # 按照训练数据的格式整理字段
    temp_result = temp_result[feature_names_new]

    prev_len = temp_result.shape[0]
    temp_result = clean_data_with_mean_std(temp_result, feature_statistics)
    after_len = temp_result.shape[0]
    # if prev_len != after_len:
    #     print("按照统计结果过滤数据前后, 数据条数: {} / {}".format(prev_len, after_len))

    # 如果过滤后的数据没有了,则跳出此次循环
    if temp_result.shape[0] == 0:
        continue

    """ personal_model 预测处理 """
    personal_sbp_pred, personal_dbp_pred = predictor.predict('personal', temp_result)
    personal_sbp_pred = remove_max_min(personal_sbp_pred)
    personal_dbp_pred = remove_max_min(personal_dbp_pred)
    
    # 添加个人的信息特征
    temp_result['Age'] = age
    temp_result['Height'] = height
    temp_result['Weight'] = weight
    temp_result['Gender'] = gender
    """ old_model 预测处理"""
    temp_result_copy = copy.deepcopy(temp_result)
    old_sbp_pred, old_dbp_pred = predictor.predict('old', temp_result_copy)
    old_sbp_pred = remove_max_min(old_sbp_pred)
    old_dbp_pred = remove_max_min(old_dbp_pred)
    
    """ kefu_model 预测处理"""
    temp_result_copy = copy.deepcopy(temp_result)
    kefu_sbp_pred, kefu_dbp_pred = predictor.predict('kefu', temp_result_copy)
    kefu_sbp_pred = remove_max_min(kefu_sbp_pred)
    kefu_dbp_pred = remove_max_min(kefu_dbp_pred)
    

    """ 添加入字段 """
    temp_result['pred_sbp_personal'] = np.mean(personal_sbp_pred)
    temp_result['pred_dbp_personal'] = np.mean(personal_dbp_pred)

    temp_result['pred_sbp_old'] = np.mean(old_sbp_pred)
    temp_result['pred_dbp_old'] = np.mean(old_dbp_pred)

    temp_result['pred_sbp_kefu'] = np.mean(kefu_sbp_pred)
    temp_result['pred_dbp_kefu'] = np.mean(kefu_dbp_pred)

    # 加入其它信息字段
    temp_result['wear_user_id'] = wear_user_id
    temp_result['date'] = data_for_test.iloc[a]['date']
    temp_result['ppg_time'] = data_for_test.iloc[a]['ppg_time']
    # 拼接到 df_results
    df_test = pd.concat([df_test, temp_result], ignore_index=True)


weight_old_model = 0.2
weight_kefu_model = 0.2
weight_personal_model = 0.6

df_test['weight_sbp'] = weight_old_model*df_test['pred_sbp_old'] + weight_kefu_model*df_test['pred_sbp_kefu'] + weight_personal_model*df_test['pred_sbp_personal']
df_test['weight_dbp'] = weight_old_model*df_test['pred_dbp_old'] + weight_kefu_model*df_test['pred_dbp_kefu'] + weight_personal_model*df_test['pred_dbp_personal']


df_test.to_csv("predict_results/{}_pred_weights01.csv".format(wear_user_id), index=False)

# 每个时间点有多组的 ppg_feature, 只保留一组, 便于查看每个时间点的模型预测值
df_test.drop_duplicates(subset=['weight_sbp', 'weight_dbp'], keep='first', ignore_index=True,  inplace=True)
df_test.to_csv("predict_results/{}_pred_weights02.csv".format(wear_user_id), index=False)