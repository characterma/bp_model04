#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(r"utils/")
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
np.random.seed(666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as LR

from andun_sql.andun_sql import AndunSql
from sklearn.metrics import mean_squared_error # MSE , RMSE对MSE开平方即可
from sklearn.metrics import mean_absolute_error # MAE
import joblib
import pickle
from utils import cal_acc, add_new_features, remove_max_min, clean_data_with_quantile_statistics, clean_data_with_mean_std
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from tqdm import tqdm


"""
对此用户的所有ppg数据进行预测, 每个时间点的 ppg 信号预测 的结果取平均, 作为此时刻的血压值
"""


def predict_all_ppg_data(wear_user_id):

    # train_data_path = "personal_models/data/{}_ppg_data.csv".format(wear_user_id)
    ppg_values_for_test = "personal_models/data/{}_ppg_history_split.csv".format(wear_user_id)

    sbp_lr = joblib.load("personal_models/saved_models/{}_lr_sbp.model".format(wear_user_id))
    dbp_lr = joblib.load("personal_models/saved_models/{}_lr_dbp.model".format(wear_user_id))
    x_std = pickle.load(open("personal_models/saved_models/{}_lr_std.pickle".format(wear_user_id), 'rb'))
    feature_statistics_path = "personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id)

    feature_statistics = pd.read_csv(feature_statistics_path, header=0, index_col=0)
    """
    对此用户的所有ppg数据进行预测, 每条ppg 信号预测一个值
    """
    # 把此用户所有的ppg信号数据进行 预测
    data_for_test = pd.read_csv(ppg_values_for_test)

    feature_names_new = FEATURE_NAME_NEW
    df_test = pd.DataFrame(columns=feature_names + ['wear_user_id', 'date', 'ppg_time', 'pred_sbp_mean', 'pred_dbp_mean'])
    # 遍历 每个 ppg_value 数据
    for a in tqdm(range(data_for_test.shape[0])):

        ppg_values = data_for_test.iloc[a]['ppg_values']
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

        prev_len = temp_result.shape[0]
        # temp_result = clean_data_with_quantile_statistics(temp_result, feature_statistics)
        temp_result = clean_data_with_mean_std(temp_result, feature_statistics)
        after_len = temp_result.shape[0]
        if prev_len != after_len:
            print("按照统计结果过滤数据前后, 数据条数: {} / {}".format(prev_len, after_len))

        # 判断删选后,是否还有数据,如果有则继续处理
        if temp_result.shape[0] > 0:
            # 归一化
            norm_temp_feature = x_std.transform(temp_result)
            # 预测
            temp_sbp_pred = sbp_lr.predict(norm_temp_feature)
            temp_sbp_pred = remove_max_min(temp_sbp_pred)
            temp_result['pred_sbp_mean'] = np.mean(temp_sbp_pred)
            temp_dbp_pred = dbp_lr.predict(norm_temp_feature)
            temp_dbp_pred = remove_max_min(temp_dbp_pred)
            temp_result['pred_dbp_mean'] = np.mean(temp_dbp_pred)

            
            # 加入其它信息字段
            temp_result['wear_user_id'] = wear_user_id
            temp_result['date'] = data_for_test.iloc[a]['date']
            temp_result['ppg_time'] = data_for_test.iloc[a]['ppg_time']

            # 拼接到 df_results
            df_test = pd.concat([df_test, temp_result], ignore_index=True)

    # 特征归一化
    feature_test = df_test[feature_names_new]
    feature_test = x_std.transform(feature_test)
    # 开始预测
    print("开始预测......")
    sbp_pred = sbp_lr.predict(feature_test)
    df_test['pred_sbp'] = sbp_pred
    dbp_pred = dbp_lr.predict(feature_test)
    df_test['pred_dbp'] = dbp_pred

    df_test.to_csv("personal_models/pred_results/{}_pred_lr_all_mean.csv".format(wear_user_id), index=False)




if __name__ == "__main__":

    wear_user_id = "902170160200952"
    predict_all_ppg_data(wear_user_id=wear_user_id)