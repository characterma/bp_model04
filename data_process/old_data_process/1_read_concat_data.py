#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import json
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
1 处理去年公司内部采集的数据, 全部使
首先按照时间 feature 的时间点 进行 train test 划分, 然后 再提取里面的feature
"""


# csv 文件数据的目录
data_dir = r"data/raw_data/data_old"
# json数据,原始的ppg信号数据
raw_data_dir = r"data/json_data/data_old"

# 合并后的数据保存路径
data_save_path = r"data_process/old_data_process/1_concat_data.csv"

feature_names = ['A1/(A1+A2)', 'A2/(A1+A2)', 'A1/AC', 'A2/AC', 'Slope',
       'DiastolicTime', 'SystolicTime', 'RR', 'Age', 'Gender', 'SBP', 'DBP',
       'Height', 'Weight', 'A1', 'A2', 'AC', 'SBP2', 'DBP2', 
       'RefDeviceID', 'AvgSBP', 'AvgDBP', 'DiffSBP', 'DiffDBP',
       'UpToMaxSlopeTime', 'HighBloodPressure', 'HRa', 'HRb','DiffHR'
       ]
# 'WearUserID', 'MedicalHistory',

# 遍历此目录,读取每个文件,合并
df_data = None

for root, dirs, files in os.walk(data_dir):
    for f_name in tqdm(files):
        f_path = os.path.join(data_dir, f_name)
        temp_data = pd.read_csv(f_path)
        # print(temp_data.columns)

        # 找出对应的此文件的json文件,提取出时间
        f_name_id = f_name.split(".")[0]
        # print("f_name_id:", f_name_id)
        f_name_json = f_name_id + ".json"
        with open(os.path.join(raw_data_dir, f_name_json), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            # print(json_data.keys())
            # print(json_data['createTime'])
            temp_data['createTime'] = json_data['createTime']

        # file_name用于后面的对train, test分开
        temp_data['file_name'] = f_name

        if isinstance(df_data, pd.core.frame.DataFrame):
            # 合并
            df_data = pd.concat([df_data, temp_data], ignore_index=True)
        else:
            df_data = temp_data
        
        # break

print(df_data.shape)
print(df_data.head())

df_data.to_csv(data_save_path, encoding='utf-8', index=False)
