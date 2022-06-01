#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(r".")
sys.path.append(r"utils/")
sys.path.append(r"personal_models/")
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
from config import Min_Minute
from tqdm import tqdm
from models.predictor import Predictor

from a_1_select_kefu_data_by_user import select_kefu_data_by_user
from a_2_user_data_analysis import pull_bp_features_by_user_from_db, bp_features_analysis
from a_3_find_feature_bp_by_user import find_feature_bp_by_user
from a_4_train_model_lr import model_lr
from a_4_train_model_rf import model_rf
from a_5_predict_all_ppg_data import predict_all_ppg_data
"""
一键添加一个使用新模型的脚本
1 通过查询数据中此用户的客服记录数据是否满足训练personal_model, 判断此是否可以加入;
2 如果此用户满足条件,则查询此用户的所有ppg信号数据, 统计特征, 利用统计特征过滤数据;
3 训练此用户的personal_model 并保存;
4 训练完成后,把此用户的相关信息保存在数据库一个单独的表中, 供java端在调用血压模型接口的时候使用, 主要是判断用户是否使用新模型
"""




if __name__ == "__main__":

    # 需要添加的用户 id
    wear_user_id = "AvKdcFjU"
    # 查询此用户的客服记录的血压值记录
    select_kefu_data_by_user(wear_user_id=wear_user_id)
    # 根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据
    find_feature_bp_by_user(wear_user_id=wear_user_id, min_minute=Min_Minute)
    # 查询此用户的 所有的血压 ppg_feature 记录值,用于统计分析
    pull_bp_features_by_user_from_db(wear_user_id=wear_user_id, last_n_day=10)
    # 统计分析此用户的 ppg_feature值 的统计分析, 用于后面的过滤数据使用
    bp_features_analysis(wear_user_id=wear_user_id)
    # 训练 personal_model
    # model_lr(wear_user_id=wear_user_id)
    model_rf(wear_user_id=wear_user_id)

    # predict_all_ppg_data(wear_user_id=wear_user_id)