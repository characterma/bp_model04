#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


bp_history = pd.read_csv("血压.csv")
bp_history = bp_history[bp_history['enabled'] == 0]
bp_history['gmt_create'] = bp_history['gmt_create'].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S") if isinstance(x, str) else datetime.strptime("2020-06-01 00:23:00", "%Y-%m-%d %H:%M:%S"))
print(bp_history.shape)
print(bp_history.head())

unique_users = bp_history['wear_user_id'].unique()
now = datetime.now()
now = now.strftime("%Y-%m-%d")
now = datetime.strptime(now, "%Y-%m-%d")
a = 1
for uu in unique_users:
    temp_history = bp_history[bp_history['wear_user_id'] == uu]

    old_len = temp_history.shape[0]

    # 删除 半年前的数据
    time_limit = now - timedelta(days=180)

    new_history = temp_history[temp_history['gmt_create'] > time_limit]

    if temp_history.shape[0] >= 10 and new_history.shape[0] < 10:

        print("{} --- {} 总的记录 {} 个， 删除半年以上的数据后，共 {} 个...".format(a, uu, temp_history.shape[0], new_history.shape[0]))

        time.sleep(1)

        a += 1
