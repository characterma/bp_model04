#!/usr/bin/python
# -*- coding: utf-8 -*-


# 一些基本配置

PORT = "50057"


# 高压,低压的限值,用于过滤极端值得数据
Max_SBP = 200
Min_SBP = 85
Max_DBP = 140
Min_DBP = 45


# 拉依达准则法 数据清洗,  std 的倍数, 使用mean-SIGMA*std  mean+SIGMA*std 
SIGMA = 3
# 使用中位数进行数据清洗
Min_Quantile = '1%'
Max_Quantile = '99%'

# 使用客服记录的数据构造训练时间时, 从客户记录的血压时间与 ppg feature 时间做匹配, 搜索的设置的时间范围
Min_Minute = 20

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# 数据库中的 ppg feature 的特征顺序
feature_names_sql = ['Slope','DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime','A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 
                     'A1', 'A2', 'AC']
                     
# nn model 的 特征顺序
feature_names_nn = ['A1/(A1+A2)','A2/(A1+A2)','A1/AC','A2/AC','Slope','DiastolicTime','SystolicTime','RR','Age','Gender','Height','Weight','UpToMaxSlopeTime']


feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
                 'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime',
                 'A1', 'A2', 'AC',
                 'Age', 'Height', 'Weight', 'AvgSBP','AvgDBP','Gender_0','Gender_1'
                 ]

# FEATURE_NAME_OLD =  ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
#                  'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime',
#                  'A1', 'A2', 'AC']
 
# FEATURE_NAME_OLD =  ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime'] # 线上 2020-09-27以前
FEATURE_NAME_OLD =  ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime', 'Slope', 'A1', 'A2', 'AC']
# FEATURE_NAME_OLD =  ['A1/AC', 'A2/AC', 'DiastolicTime', 'SystolicTime', 'RR', 'UpToMaxSlopeTime', 'Slope']

# DT/ST = DiastolicTime / SystolicTime
# DT-ST = DiastolicTime - SystolicTime
# A1/RR = A1/RR
# A2/RR = A2/RR
# UT/ST = UpToMaxSlopeTime / SystolicTime
# UT/RR = UpToMaxSlopeTime / RR
# ST/RR = SystolicTime / RR
# DT/RR = DiastolicTime / RR
# ST*AC = SystolicTime * AC
# A1/(ST*AC) = A1/(SystolicTime*AC)
# DT*AC = DiastolicTime * AC
# A2/(DT*AC) = A2/(DiastolicTime*AC)
# 2020-09-27 新加入
# AC/ST = AC/SystolicTime
# AC/DT = AC/DiastolicTime
# 
# NEW_FEATURE = ['A1-A2', 'DT/ST', 'DT-ST', 'A1/RR', 'A2/RR', 'UT/ST', 'UT/RR', 'ST/RR', 'DT/RR', 'ST*AC', 'A1/(ST*AC)', 'DT*AC', 'A2/(DT*AC)'] 

# 只使用此值特征, 幅值特征受干扰比较大
# 2020-09-19  去除 (A1-A2)/(A1+A2), 'A1/(A1+A2)', 'A2/(A1+A2)',
# NEW_FEATURE = ['DT/ST', 'UT/ST', 'UT/RR', 'ST/RR', 'DT/RR', 'A1/(ST*AC)', 'A2/(DT*AC)'] 
NEW_FEATURE = ['(A1-A2)/(A1+A2)', 'DT/ST', 'UT/ST', 'UT/RR', 'ST/RR', 'DT/RR', 'A1/(ST*AC)', 'A2/(DT*AC)', 'Volume', 'Volume/Slope', 'Volume/RR'] # 线上
FEATURE_NAME_NEW = FEATURE_NAME_OLD + NEW_FEATURE


old_model_weight = 0.2
kefu_model_weight = 0.2
personal_model_weight = 0.6


# last_n_month 表示从当前时刻起,往前推 几个月份, 就是对于某个用户只使用最近 若干个月的ppg_feature数据做统计分析
Last_N_Day = 90



# 线性回归模型 回归系数保存文件
Coef_File_Path = "personal_models/coef_results/lr_model_coef_data.csv"



# 血压调整
time_interval = [4, 8, 16, 30]
limit_percentage = [0.05, 0.1, 0.15, 0.2]