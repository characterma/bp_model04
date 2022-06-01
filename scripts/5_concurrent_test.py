#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
# sys.path.append(r"old_model/")
sys.path.append(r".")
sys.path.append(r"utils/")
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(r'grpc/proto/')
import numpy as np
import pandas as pd
import grpc
import anbp_pb2, anbp_pb2_grpc

from threading import Thread
import requests
import time


"""
血压模型并发测试
"""


# 线上 NG 的地址
IP = "39.97.198.78:50057"
# IP = "39.96.40.181:50057"

# 旧模型 NG 的地址
# IP = "47.95.230.250:50056"
# IP = "39.102.37.213:50056"



test_bp_features = "6.092850,0.453125,0.273438,0.726562,0.152344,0.460254,0.539746,0.152789,0.181171,925.855591,1084.721069,6018.980469,6.186612,0.449219,0.269531,0.718750,0.152344,0.459619,0.540381,0.147882,0.175636,880.776978,1055.958984,5983.052734,6.103845,0.451172,0.269531,0.726562,0.152344,0.458701,0.541299,0.148643,0.180696,814.927002,970.325439,5452.737305,6.180950,0.455078,0.269531,0.724609,0.152344,0.456755,0.543245,0.148643,0.181841,878.679321,1052.084961,5626.586426,6.183826,0.458984,0.269531,0.724609,0.152344,0.461689,0.538311,0.150072,0.178924,992.578979,1119.998901,6335.895508,6.097229,0.462891,0.273438,0.738281,0.158203,0.459264,0.540736,0.150417,0.177865,1006.262695,1145.663208,6423.263672,6.096413,0.462891,0.277344,0.738281,0.156250,0.459264,0.540736,0.149238,0.176603,1005.657471,1189.702759,6522.362793,6.018370,0.470703,0.277344,0.750000,0.158203,0.457386,0.542614,0.153394,0.181474,1037.984619,1249.816406,6888.174805,5.991086,0.470703,0.283203,0.750000,0.162109,0.457371,0.542629,0.154491,0.187520,1043.498169,1289.709473,7004.988281"

user_info = """{
    "wearUserId": "6617ffbe",
    "age": 33,
    "height": 182,
    "weight": 70,
    "gender": 1
}"""


def run(num):
    channel = grpc.insecure_channel(IP)
    stub = anbp_pb2_grpc.GreeterStub(channel=channel)
    response = stub.ANBP(anbp_pb2.BPRequest(userInfo = user_info,
                                             oStatus = "1",
                                             features=test_bp_features,
                                             preReportBP="150,90,90",
                                            #  preReportBP="0,-1",
                                             preReportTimeDiffMinute = "31"))
    print("num: {}, status: {}, bloodPpressure: {}, timestamp: {}".format(num, response.status, response.bloodPpressure, response.timestamp))

# 并发数
NUM_REQUESTS = 500
# 请求间隔
SLEEP_COUNT = 0.05

for i in range(0, NUM_REQUESTS):
    # 创建线程
    t = Thread(target=run, args=[i])
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

time.sleep(300)