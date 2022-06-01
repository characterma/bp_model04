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

from sklearn.metrics import mean_squared_error # MSE , RMSE对MSE开平方即可
from sklearn.metrics import mean_absolute_error # MAE
# from sklearn.externals import joblib
import joblib
import pickle
from utils import cal_acc



# train_data_path = r"data/train_data/train_data_old.csv"
train_data_path = r"data/train_data/train_data_kefu.csv"


test_data_path = r"data/test_data/test_data_old.csv"
# test_data_path = r"data/test_data/test_data_kefu.csv"

# 读取数据
train_data = pd.read_csv(train_data_path, encoding='utf-8')
test_data = pd.read_csv(test_data_path, encoding='utf-8')

# 旧版本使用的特征, model 1,2,3,4,5,6
feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
                 'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime', 
                 'A1', 'A2', 'AC',
                 'Age', 'Height', 'Weight']

label_names = ['AvgSBP', 'AvgDBP']

train_x = train_data[feature_names]
train_y = train_data[label_names]
test_x = test_data[feature_names]
test_y = test_data[label_names]


# 对x 进行归一化处理
x_std = StandardScaler()
train_x = x_std.fit_transform(train_x)
test_x = x_std.transform(test_x)

# 保存归一化变量
pickle.dump(x_std, open(r"models/saved_models/rf_std.pickle", 'wb'))

# 把one hot 编码的 Gender 特征 添加
if 'Gender_0' not in train_data.columns:
    train_data = pd.get_dummies(train_data, columns=['Gender'])
train_x = np.concatenate([train_x, train_data[['Gender_0', 'Gender_1']].values], axis=1)
# train_x = np.concatenate([train_x, pd.get_dummies(train_data['Gender']).values], axis=1)

if 'Gender_0' not in test_data.columns:
    test_data = pd.get_dummies(test_data, columns=['Gender'])
test_x = np.concatenate([test_x, test_data[['Gender_0', 'Gender_1']].values], axis=1)



""" 
建模,训练
"""
# 创建可选参数字典
tree_param_grid = { 
                    'n_estimators':[120],
                    'min_samples_split': [2],
                    'max_depth': [35],
                    'min_samples_leaf': [1],
                    'random_state': [66]
                    }

print("开始训练 SBP 模型...")

sbp_rfr = RandomForestRegressor(
                                # min_samples_split = 2,
                                # n_estimators = 120,
                                # max_depth = 35,
                                # min_samples_leaf = 1,
                                # random_state = 66
                                )

# 创建选参对象，传入模型对象，参数字典，以及指定进行5折交叉验证
sbp_grid = GridSearchCV(sbp_rfr, param_grid=tree_param_grid, cv=5)
# 向选参对象传入训练集数据
sbp_grid.fit(train_x, train_y['AvgSBP'])
sbp_pred = sbp_grid.predict(test_x)
# 打印各参数组合得分
print(sbp_grid.cv_results_['mean_test_score'])
# 打印最佳参数及其得分
print(sbp_grid.best_params_, sbp_grid.best_score_)


sbp_grid.fit(train_x, train_y['AvgSBP'])
sbp_pred = sbp_grid.predict(test_x)
print("SBP mean_squared_error MSE:", mean_squared_error(test_y['AvgSBP'].values, sbp_pred))
print("SBP mean_absolute_error MAE:", mean_absolute_error(test_y['AvgSBP'].values, sbp_pred))
cal_acc(test_y['AvgSBP'].values, sbp_pred)

# 对第一批的训练数据进行测试
# print("保存模型前, 对第一批的训练数据进行测试.......")
# sbp_pred = sbp_grid.predict(train_x)
# print("SBP mean_squared_error MSE:", mean_squared_error(train_y['AvgSBP'].values, sbp_pred))
# print("SBP mean_absolute_error MAE:", mean_absolute_error(train_y['AvgSBP'].values, sbp_pred))
# cal_acc(train_y['AvgSBP'].values, sbp_pred)




# 保存模型
joblib.dump(sbp_rfr, r"models/saved_models/rfr_sbp.model", compress=3)

# 模型再训练
# print("对 rfr 模型进行再训练......")
# sbp_rfr = joblib.load(r"models/saved_models/rfr_sbp.model")
# sbp_rfr.fit(test_x, test_y['AvgSBP'])
# sbp_pred = sbp_rfr.predict(test_x)

# print("SBP mean_squared_error MSE:", mean_squared_error(test_y['AvgSBP'].values, sbp_pred))
# print("SBP mean_absolute_error MAE:", mean_absolute_error(test_y['AvgSBP'].values, sbp_pred))
# cal_acc(test_y['AvgSBP'].values, sbp_pred)

# 对第一批的训练数据进行测试
# print("对第一批的训练数据进行测试.......")
# sbp_pred = sbp_rfr.predict(train_x)
# print("SBP mean_squared_error MSE:", mean_squared_error(train_y['AvgSBP'].values, sbp_pred))
# print("SBP mean_absolute_error MAE:", mean_absolute_error(train_y['AvgSBP'].values, sbp_pred))
# cal_acc(train_y['AvgSBP'].values, sbp_pred)




print("SBP 模型训练结束, 开始训练 DBP模型")

dbp_rfr = RandomForestRegressor(
                                # min_samples_split = 2,
                                # n_estimators = 120,
                                # max_depth = 35,
                                # min_samples_leaf = 1,
                                # random_state = 66
                                )

# 创建选参对象，传入模型对象，参数字典，以及指定进行5折交叉验证
dbp_grid = GridSearchCV(dbp_rfr, param_grid=tree_param_grid, cv=5)
# 向选参对象传入训练集数据
dbp_grid.fit(train_x, train_y['AvgDBP'])
dbp_pred = dbp_grid.predict(test_x)
# 打印各参数组合得分
print(dbp_grid.cv_results_['mean_test_score'])
# 打印最佳参数及其得分
print(dbp_grid.best_params_, dbp_grid.best_score_)


# 向选参对象传入训练集数据
dbp_grid.fit(train_x, train_y['AvgDBP'])
dbp_pred = dbp_grid.predict(test_x)


print("DBP mean_squared_error MSE:", mean_squared_error(test_y['AvgDBP'].values, dbp_pred))
print("DBP mean_absolute_error MAE:", mean_absolute_error(test_y['AvgDBP'].values, dbp_pred))
cal_acc(test_y['AvgDBP'].values, dbp_pred)


# 保存模型
# joblib.dump(dbp_rfr, r"models/saved_models/rfr_dbp.model", compress=3)


# 保存预测的 sbp, dbp
test_data['pred_sbp'] = sbp_pred
test_data['sbp_diff'] = abs(test_data['pred_sbp'] - test_data['AvgSBP'])
test_data['pred_dbp'] = dbp_pred
test_data['dbp_diff'] = abs(test_data['pred_dbp'] - test_data['AvgDBP'])

# test_data.to_csv(r"predict_results/test_data_kefu_predict_rf.csv", encoding='utf-8', index=False)