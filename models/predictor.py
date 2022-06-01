#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import copy
import os
import sys
sys.path.append(".")
sys.path.append(r"utils/")
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
np.random.seed(666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as LR

from andun_sql.andun_sql import AndunSql
import joblib
import pickle
from utils import cal_acc, add_new_features, remove_max_min
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names
from config import Max_SBP, Min_SBP, Max_DBP, Min_DBP, old_model_weight, kefu_model_weight, personal_model_weight
from tqdm import tqdm

from log import MyLogger

bp_predictor_log = MyLogger('./log/predictor.log', level='info')


model_type = 'rfr'
model_version = 'new'

# 使用公司去年内部采集的数据训练的model
model_old_path = {
    'sbp': "models/saved_models/{}_sbp_old_{}.model".format(model_type, model_version),
    'dbp': "models/saved_models/{}_dbp_old_{}.model".format(model_type, model_version),
    'std': "models/saved_models/rfr_std_old_{}.pickle".format(model_version)
}
# 从数据库里,客服记录的用户血压值训练的model
model_kefu_path = {
    'sbp': "models/saved_models/{}_sbp_kefu_{}.model".format(model_type, model_version),
    'dbp': "models/saved_models/{}_dbp_kefu_{}.model".format(model_type, model_version),
    'std': "models/saved_models/rfr_std_kefu_{}.pickle".format(model_version)
}
# 每个用户单独训练的自己的model, 线性回归模型
model_personal_path = {
    'sbp': "personal_models/saved_models/{}_lr_sbp.model",
    'dbp': "personal_models/saved_models/{}_lr_dbp.model",
    'std': "personal_models/saved_models/{}_lr_std.pickle"
}



# 用于加载模型并预测
class Predictor():


    def __init__(self, wear_user_id):

        self.wear_user_id = wear_user_id
        # 加载模型
        self.model_old_sbp = joblib.load(model_old_path['sbp'])
        self.model_old_dbp = joblib.load(model_old_path['dbp'])
        self.old_std = joblib.load(model_old_path['std'])

        self.model_kefu_sbp = joblib.load(model_kefu_path['sbp'])
        self.model_kefu_dbp = joblib.load(model_kefu_path['dbp'])
        self.kefu_std = joblib.load(model_kefu_path['std'])

        try:
            self.model_personal_sbp = joblib.load(model_personal_path['sbp'].format(wear_user_id))
            self.model_personal_dbp = joblib.load(model_personal_path['dbp'].format(wear_user_id))
            self.personal_std = joblib.load(model_personal_path['std'].format(wear_user_id))
        except Exception as e:
            bp_predictor_log.logger.error(e)
            bp_predictor_log.logger.error("{} 加载此用户的personal_model失败...".format(self.wear_user_id))

        # user_info, 此用户的最高,最低血压,及每个model 的 weight
        try:
            self.ansql = AndunSql()
            user_info = self.ansql.ansql_user_info_with_new_model(wear_user_id=wear_user_id)
            self.Max_SBP = int(user_info['max_sbp'].tolist()[0])
            self.Min_SBP = int(user_info['min_sbp'].tolist()[0])
            self.Max_DBP = int(user_info['max_dbp'].tolist()[0])
            self.Min_DBP = int(user_info['min_dbp'].tolist()[0])

            self.old_model_weight = float(user_info['old_model_weight'].tolist()[0])
            self.kefu_model_weight = float(user_info['kefu_model_weight'].tolist()[0])
            self.personal_model_weight = float(user_info['personal_model_weight'].tolist()[0])
        except Exception as e:

            self.Max_SBP = Max_SBP
            self.Min_SBP = Min_SBP
            self.Max_DBP = Max_DBP
            self.Min_DBP = Min_DBP
            self.old_model_weight = old_model_weight
            self.kefu_model_weight = kefu_model_weight
            self.personal_model_weight = personal_model_weight
            bp_predictor_log.logger.error(e)
            bp_predictor_log.logger.error("{} 加载此用户的andun_watch.d_users_bp_model信息失败...".format(self.wear_user_id))


    # 预测
    def predict(self, which_model, data):

        if which_model == 'old':
            data = self.std_process('old', data)
            pred_sbp = self.model_old_sbp.predict(data)
            pred_dbp = self.model_old_dbp.predict(data)
        elif which_model == 'kefu':
            data = self.std_process('kefu', data)
            pred_sbp = self.model_kefu_sbp.predict(data)
            pred_dbp = self.model_kefu_dbp.predict(data)
        elif which_model == 'personal':
            data = self.personal_std.transform(data)
            pred_sbp = self.model_personal_sbp.predict(data)
            pred_dbp = self.model_personal_dbp.predict(data)
            pred_sbp, pred_dbp = self.min_max_process(pred_sbp, pred_dbp)

        else:
            print("Predictor...选择的模型不对...")
            pred_dbp = None
            pred_sbp = None

        return pred_sbp, pred_dbp

    # 归一化处理
    def std_process(self, which_model, data):
        
        gender = data['Gender'].tolist()[0]
        if gender == 1:
            data['Gender_0'] = 0
            data['Gender_1'] = 1
        else:
            data['Gender_0'] = 1
            data['Gender_1'] = 0

        df_gender = data[['Gender_0', 'Gender_1']]

        data.drop(columns=['Gender', 'Gender_0', 'Gender_1'], axis=1, inplace=True)
        if which_model == 'old':
            data = self.old_std.transform(data)
        elif which_model == 'kefu':
            data = self.kefu_std.transform(data)

        data = np.concatenate([data, df_gender.values], axis=1)

        return data

    # 对预测结果的极端值进行处理, 主要是针对于 personal_model
    # 加入一个 random(),是避免 当一组预测值都很大的时候,按照 Max, 或者Min截取, 后面做最大最小值过滤的时候,防止把所有最大值,或者最小值都删除了
    def min_max_process(self, pred_sbp, pred_dbp):
        pred_sbp = [p if p<=self.Max_SBP else self.Max_SBP+random.random() for p in pred_sbp]
        pred_sbp = [p if p>=self.Min_SBP else self.Min_SBP+random.random() for p in pred_sbp]

        pred_dbp = [p if p<=self.Max_DBP else self.Max_DBP+random.random() for p in pred_dbp]
        pred_dbp = [p if p>=self.Min_DBP else self.Min_DBP+random.random() for p in pred_dbp]

        return pred_sbp, pred_dbp


    def predict_with_models_weight(self, data, userInfo):

        """ personal_model 预测处理 """
        personal_sbp_pred, personal_dbp_pred = self.predict('personal', data)
        personal_sbp_pred = remove_max_min(personal_sbp_pred)
        personal_dbp_pred = remove_max_min(personal_dbp_pred)
        # bp_server_log.logger.info("{} personal_model 预测 完毕...".format(self.wear_user_id))
        # 添加个人的信息特征
        data['Age'] = float(userInfo['age'])
        data['Height'] = float(userInfo['height'])
        data['Weight'] = float(userInfo['weight'])
        data['Gender'] = int(userInfo['gender'])
        """ old_model 预测处理"""
        temp_result_copy = copy.deepcopy(data)
        old_sbp_pred, old_dbp_pred = self.predict('old', temp_result_copy)
        old_sbp_pred = remove_max_min(old_sbp_pred)
        old_dbp_pred = remove_max_min(old_dbp_pred)
        # bp_server_log.logger.info("{} old_model 预测 完毕...".format(self.wear_user_id))
        
        """ kefu_model 预测处理"""
        temp_result_copy = copy.deepcopy(data)
        kefu_sbp_pred, kefu_dbp_pred = self.predict('kefu', temp_result_copy)
        kefu_sbp_pred = remove_max_min(kefu_sbp_pred)
        kefu_dbp_pred = remove_max_min(kefu_dbp_pred)
        # bp_server_log.logger.info("{} kefu_model 预测 完毕...".format(self.wear_user_id))
        

        """ 计算均值 """
        pred_sbp_personal = np.mean(personal_sbp_pred)
        pred_dbp_personal = np.mean(personal_dbp_pred)

        pred_sbp_old = np.mean(old_sbp_pred)
        pred_dbp_old = np.mean(old_dbp_pred)

        pred_sbp_kefu = np.mean(kefu_sbp_pred)
        pred_dbp_kefu = np.mean(kefu_dbp_pred)

        SBP = self.old_model_weight*pred_sbp_old + self.kefu_model_weight*pred_sbp_kefu + self.personal_model_weight*pred_sbp_personal
        DBP = self.old_model_weight*pred_dbp_old + self.kefu_model_weight*pred_dbp_kefu + self.personal_model_weight*pred_dbp_personal

        bp_predictor_log.logger.info("{} 模型预测值: pred_sbp_old:{}, pred_dbp_old:{}, pred_sbp_kefu:{}, pred_dbp_kefu:{}, pred_sbp_personal:{}, pred_dbp_personal:{}".format(self.wear_user_id, int(pred_sbp_old),int(pred_dbp_old),int(pred_sbp_kefu),int(pred_dbp_kefu),int(pred_sbp_personal),int(pred_dbp_personal)))

        return SBP, DBP