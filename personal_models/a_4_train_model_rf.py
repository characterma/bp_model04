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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from andun_sql.andun_sql import AndunSql
from sklearn.metrics import mean_squared_error # MSE , RMSE对MSE开平方即可
from sklearn.metrics import mean_absolute_error # MAE
import joblib
import pickle
from utils import cal_acc, add_new_features, clean_data_with_quantile_statistics, clean_data_with_mean_std
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from config import Coef_File_Path



def model_rf(wear_user_id):

    train_data_path = "personal_models/data/{}_ppg_data.csv".format(wear_user_id)
    # ppg_values_for_test = "personal_models/data/{}_ppg_history_split.csv".format(wear_user_id)
    feature_statistics_path = "personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id)

    # feature_statistics = pd.read_csv(feature_statistics_path, header=0, index_col=0)
    # 使用单独用户自己的数据训练 model
    train_data = pd.read_csv(train_data_path)
    # 去除nan
    train_data = train_data[train_data['AC'] > 0.0]
    train_data = train_data[train_data['A1'] > 0.0]
    train_data = train_data[train_data['A2'] > 0.0]
    train_data.dropna(axis=0, how="any", inplace=True)
    # 使用新增加的特征
    # feature_names = FEATURE_NAME_NEW + ['Age', 'Height', 'Weight']
    feature_names_new = FEATURE_NAME_NEW
    # 构造新特征 ['A1-A2', 'DT/ST', 'DT-ST', 'A1/RR', 'A2/RR', 'UT/ST','UT/RR', 'ST/RR', 'DT/RR', 'ST*AC', 'A1/(ST*AC)', 'DT*AC', 'A2/(DT*AC)'] 
    train_data = add_new_features(train_data)
    # print("按照统计结果过滤数据前, 数据条数: {}".format(train_data.shape))
    # train_data = clean_data_with_statistics(train_data, feature_statistics)
    # train_data = clean_data_with_mean_std(train_data, feature_statistics)
    print("按照统计结果过滤数据后, 数据条数: {}".format(train_data.shape))

    feature_train_data = train_data[feature_names_new]

    # 对x 进行归一化处理
    x_std = StandardScaler()
    # feature_train_data.to_csv(r"C:\Users\jianbin.xu\Desktop\Andun\test_nan_too_large.csv")
    feature_train_data = x_std.fit_transform(feature_train_data)
    # test_x = x_std.transform(test_x)
    # 保存归一化变量
    pickle.dump(x_std, open("personal_models/saved_models/{}_lr_std.pickle".format(wear_user_id), 'wb'))

    print("开始训练 SBP 模型...")
    sbp_rfr = RandomForestRegressor()

    # 一 直接训练
    sbp_rfr.fit(feature_train_data, train_data['sbp'])
    # 二 使用交叉验证
    # sbp_scores = cross_val_score(sbp_rfr, feature_train_data, train_data['sbp'], cv=5)
    # print("sbp train scores: {}".format(sbp_scores))

    sbp_pred = sbp_rfr.predict(feature_train_data)
    train_data['pred_sbp'] = sbp_pred

    print("SBP mean_squared_error MSE:", mean_squared_error(train_data['sbp'].values, sbp_pred))
    print("SBP mean_absolute_error MAE:", mean_absolute_error(train_data['sbp'].values, sbp_pred))
    cal_acc(train_data['sbp'].values, sbp_pred)
    # 打印特征重要性
    print(sorted(zip(map(lambda x: round(x, 4), sbp_rfr.feature_importances_), feature_names_new), reverse=True))
    # 保存模型
    joblib.dump(sbp_rfr, "personal_models/saved_models/{}_lr_sbp.model".format(wear_user_id))




    print("开始训练 DBP 模型...")
    dbp_rfr = RandomForestRegressor()

    # 一 直接训练
    dbp_rfr.fit(feature_train_data, train_data['dbp'])
    # 二 使用交叉验证
    # dbp_scores = cross_val_score(dbp_rfr, feature_train_data, train_data['dbp'], cv=5)
    # print("dbp train scores: {}".format(dbp_scores))

    dbp_pred = dbp_rfr.predict(feature_train_data)
    train_data['pred_dbp'] = dbp_pred

    print("DBP mean_squared_error MSE:", mean_squared_error(train_data['dbp'].values, dbp_pred))
    print("DBP mean_absolute_error MAE:", mean_absolute_error(train_data['dbp'].values, dbp_pred))
    cal_acc(train_data['dbp'].values, dbp_pred)

    # 打印特征重要性
    print(sorted(zip(map(lambda x: round(x, 4), dbp_rfr.feature_importances_), feature_names_new), reverse=True))

    # 保存模型
    joblib.dump(dbp_rfr, "personal_models/saved_models/{}_lr_dbp.model".format(wear_user_id))

    # 保存预测的结果
    train_data.to_csv("personal_models/pred_results/{}_pred_rf.csv".format(wear_user_id), index=False)


    # 保存 线性回归模型的 回归系数
    
    # if os.path.exists(Coef_File_Path):
    #     df_coef = pd.read_csv(Coef_File_Path)
    # else:
    #     df_coef = pd.DataFrame(columns=feature_names_new + ['wear_user_id', 'bp_type'])

    # df_sbp_coef = pd.DataFrame(data=sbp_lr.coef_.reshape(1,len(feature_names_new)), columns=feature_names_new)
    # df_sbp_coef['wear_user_id'] = wear_user_id
    # df_sbp_coef['bp_type'] = 'sbp'
    # df_dbp_coef = pd.DataFrame(data=dbp_lr.coef_.reshape(1,len(feature_names_new)), columns=feature_names_new)
    # df_dbp_coef['wear_user_id'] = wear_user_id
    # df_dbp_coef['bp_type'] = 'dbp'

    # df_coef = pd.concat([df_coef, df_sbp_coef], ignore_index=True)
    # df_coef = pd.concat([df_coef, df_dbp_coef], ignore_index=True)

    # # 去重
    # # df_coef.drop_duplicates(subset=['wear_user_id', 'bp_type'], keep='last', ignore_index=True,  inplace=True)
    # # 保存
    # df_coef.to_csv(Coef_File_Path, index=False)

if __name__ == "__main__":

    wear_user_id = "902170160200952"

    model_rf(wear_user_id=wear_user_id)
