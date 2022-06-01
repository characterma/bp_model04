#!/usr/bin/python
# -*- coding: utf-8 -*-
from concurrent import futures
import time
import grpc
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(".")
sys.path.append(r'utils/')
sys.path.append(r'grpc/')
sys.path.append(r"grpc/proto")
sys.path.append(r"personal_models")
sys.path.append(r"nn_models")
from config import NEW_FEATURE, FEATURE_NAME_NEW, FEATURE_NAME_OLD, feature_names_sql, feature_names, feature_names_nn
from config import Min_Minute, _ONE_DAY_IN_SECONDS, Last_N_Day, PORT
from utils import cal_acc, add_new_features, remove_max_min, clean_data_with_quantile_statistics, clean_data_with_mean_std, check_ppg_feature_vaild
from utils import adjust_bp
from log import MyLogger
import anbp_pb2    # 消息
import anbp_pb2_grpc   # 服务
import numpy as np
import json
import datetime
import ssl
import os
from models.predictor import Predictor
import pandas as pd
import copy
from keras.models import load_model

from a_1_select_kefu_data_by_user import select_kefu_data_by_user
from a_2_user_data_analysis import pull_bp_features_by_user_from_db, bp_features_analysis
from a_3_find_feature_bp_by_user import find_feature_bp_by_user
from a_4_train_model_lr import model_lr
from a_4_train_model_rf import model_rf
from a_5_predict_all_ppg_data import predict_all_ppg_data

from deep_learning_model import Deeplearningmodel
from utils import send_ding_message, is_in_wx_model


# 返回状态码定义
ANBP_RES_FAILD = '0'
ANBP_RES_SUCCEED = '1'
ANBP_TRAIN_SUCCESS = '2'
ANBP_RES_L_ABNORMAL_BP = '3'
ANBP_TRAIN_FAILED = '4'
ANBP_TRAIN_NO_MODEL = '5'
ANBP_USER_IN_WX_MODEL = '6'

# grpc_server的日志
bp_server_log = MyLogger('./log/server.log', level='info')
# 训练模型的日志
bp_train_log = MyLogger('./log/train.log', level='info')
# 神经网络模型的预测日志
bp_nn_log = MyLogger('./log/nn.log', level='info')

# 导入模型并固化
# nn_model = Deeplearningmodel()

bp_server_log.logger.info('开始加载 nn model......')
nn_model_ = load_model('nn_models/models/common.h5')
temp = pd.read_csv('grpc/server/temp.csv')
x = temp[:][['A1/(A1+A2)','A2/(A1+A2)','A1/AC','A2/AC','Slope','DiastolicTime','SystolicTime','RR','Age','Gender','Height','Weight','UpToMaxSlopeTime']]
X_test = x.values
nn_model_dict = {'nn_model': nn_model_}
print('NN_Model is load, Test value is {}'.format(nn_model_dict['nn_model'].predict(X_test)))

bp_server_log.logger.info('nn model初始化完成...')





try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


#####################################
class Greeter(anbp_pb2_grpc.GreeterServicer):
    # def __init__(self):
    #     self.nn_model = Deeplearningmodel()
    def ANBP(self, request, context):
        # 状态码
        oStatus = request.oStatus
        # 用户基本信息, 身高、性别、年龄、体重，再增加一个 wear_user_id
        userInfo = request.userInfo
        userInfo = json.loads(userInfo)
        bp_server_log.logger.info("wearUserId: {}, age: {}, height: {}, weight: {}".format(userInfo['wearUserId'], userInfo['age'], userInfo['height'], userInfo['weight']))
        wear_user_id = userInfo['wearUserId']

        # 判断此用户是否在武轩的模型中
        res = is_in_wx_model(userInfo['wearUserId'])
        if res == 1:
            bp_server_log.logger.error('{} 也在武轩的模型中'.format(wear_user_id))
            # 发送消息到钉钉群
            send_ding_message('wear_user_id: {} --> 也在武轩的模型中'.format(wear_user_id))
            return anbp_pb2.BPReply(status=ANBP_USER_IN_WX_MODEL)

        # 判断 oStatus 的值,是不是 -1, 是的话说明是需要对 wear_user_id 进行建模
        if oStatus in ["-1", "-2"]:
            try:
                bp_train_log.logger.info("{} 开始训练模型...查询此用户的客服记录的血压值记录".format(wear_user_id))
                # 查询此用户的客服记录的血压值记录
                select_kefu_data_by_user(wear_user_id=wear_user_id)
                # bp_train_log.logger.info("{} 查询此用户的 所有的血压 ppg_feature 记录值,用于统计分析".format(wear_user_id))
                # 查询此用户的 所有的血压 ppg_feature 记录值,用于统计分析
                # pull_bp_features_by_user_from_db(wear_user_id=wear_user_id, last_n_day=Last_N_Day)
                # bp_train_log.logger.info("{} 统计分析此用户的 ppg_feature值 的统计分析, 用于后面的过滤数据使用".format(wear_user_id))
                # 统计分析此用户的 ppg_feature值 的统计分析, 用于后面的过滤数据使用
                # bp_features_analysis(wear_user_id=wear_user_id)
                bp_train_log.logger.info("{} 根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据".format(wear_user_id))
                # 根据客服记录的用户测量的每次的血压时间, 找出同一天手环 里面最接近 对应时间点的ppg信号数据
                find_feature_bp_by_user(wear_user_id=wear_user_id, min_minute=Min_Minute)
                bp_train_log.logger.info("{} 训练 personal_model".format(wear_user_id))
                # 训练 personal_model
                if oStatus == "-1":
                    model_lr(wear_user_id=wear_user_id)
                    return anbp_pb2.BPReply(status=ANBP_TRAIN_SUCCESS)
                elif oStatus == "-2":
                    model_rf(wear_user_id=wear_user_id)
                    return anbp_pb2.BPReply(status=ANBP_TRAIN_SUCCESS)
                else:
                    return anbp_pb2.BPReply(status=ANBP_TRAIN_NO_MODEL) 
            except Exception as e:
                print(e)
                bp_train_log.logger.error("{} 训练 personal_model 发生错误......".format(wear_user_id))
                return anbp_pb2.BPReply(status=ANBP_TRAIN_FAILED)

        # 执行预测
        try:
            # ppg 特征
            features = request.features
            # 上次预测的血压值
            preReportBP = request.preReportBP

            # 检查ppg feature置信度
            ppg_ci, features = check_ppg_feature_vaild(features)
            bp_server_log.logger.info("  ppg confidence interval: {}".format(str(ppg_ci)))
            if 0 == ppg_ci:
                return anbp_pb2.BPReply(status=ANBP_RES_FAILD)

            first_bp_flag = False
            previousSBP, previousDBP, previousCi = [0, 0, -1]
            preReportTimeDiffMinute = 0

            if preReportBP == '0,-1':
                first_bp_flag = True
            else:
                previousSBP, previousDBP, previousCi = [int(t) for t in preReportBP.split(',')]
                preReportTimeDiffMinute = int(request.preReportTimeDiffMinute)
            bp_server_log.logger.info('wear_user_id: {} | Status num: {} | pre Report BP: previousSBP:{} / previousDBP:{} / previousCi: {} | preReportTimeDiffMinute:{}'.format(wear_user_id, oStatus, previousSBP, previousDBP, previousCi, preReportTimeDiffMinute))
        except Exception as e:
            bp_server_log.logger.info(e)
            return anbp_pb2.BPReply(status=ANBP_RES_FAILD)


        try:
            bp_server_log.logger.info("{} 开始预测...".format(wear_user_id))
            # 封装的预测类
            predictor = Predictor(wear_user_id=wear_user_id)
            # 用于模型的特征顺序
            feature_names_new = FEATURE_NAME_NEW
            # feature_statistics_path = "personal_models/data/{}_bp_features_analysis.csv".format(wear_user_id)
            # feature_statistics = pd.read_csv(feature_statistics_path, header=0, index_col=0)
            # bp_server_log.logger.info("{} 加载 feature_statistics 完毕...".format(wear_user_id))

            # ppg feature的分组整理
            ppg_values = features
            # ppg_values = list(ppg_values.split(','))
            ppg_values = np.array(ppg_values)
            ppg_values = ppg_values.reshape(-1, 12)

            # 把ppg_values保存成 dataframe
            temp_result = pd.DataFrame(data=ppg_values, columns=feature_names_sql)
            # 过滤掉为0的ppg特征值
            temp_result = temp_result[~(temp_result == 0.0).any(axis=1)]
            temp_result = temp_result.astype('float')

            # 增加新特征
            temp_result = add_new_features(temp_result)
            # 按照训练数据的格式整理字段
            temp_result = temp_result[feature_names_new]
            bp_server_log.logger.info("{} 增加新特征 完毕...".format(wear_user_id))

            # prev_len = temp_result.shape[0]
            # temp_result = clean_data_with_mean_std(temp_result, feature_statistics)
            # after_len = temp_result.shape[0]
            # bp_server_log.logger.info("{} 过滤数据 完毕...".format(wear_user_id))

            # 如果过滤后的数据没有了,则返回 sbp, dbp 为空
            if temp_result.shape[0] < 3:
                # bp_server_log.logger.warn("{} 的ppg_feature 本次传入的数据都被过滤了,不执行预测...".format(wear_user_id))
                bp_server_log.logger.warn("{} 的ppg_feature {} 组,不执行预测...".format(wear_user_id, temp_result.shape[0]))
                SBP = None
                DBP = None
                return anbp_pb2.BPReply(status=ANBP_RES_FAILD, bloodPpressure= '0' + '/' + '0' + '/' +str(ppg_ci))
            else:
                # """ personal_model 预测处理 """
                SBP, DBP = predictor.predict_with_models_weight(temp_result, userInfo)

            bp_server_log.logger.info("{} 经过权重输出,没有经过血压调整: SBP:{}, DBP:{}".format(wear_user_id, round(SBP), round(DBP)))

            bp_nn_log.logger.info("{}: ml model的预测值: SBP:{}, DBP:{}".format(wear_user_id, round(SBP), round(DBP)))
            
            # 如果是第一次预测, 直接返回结果
            if first_bp_flag:
                bp_result = str(round(SBP)) + '/' + str(round(DBP))
                return anbp_pb2.BPReply(status=ANBP_RES_SUCCEED, bloodPpressure=bp_result+'/'+str(ppg_ci))
            # 和前一次的血压值相比较, 短时间内变化幅度大的,需要做调整
            else:
                SBP, DBP = adjust_bp(SBP, DBP, previousSBP, previousDBP, preReportTimeDiffMinute)
                bp_server_log.logger.info("{} 经过血压调整后: SBP:{}, DBP:{}".format(wear_user_id, SBP, DBP))
                if DBP >= SBP:
                    bp_server_log.logger.error('{} DBP >= SBP ,skip it!'.format(wear_user_id))
                    return anbp_pb2.BPReply(status=ANBP_RES_L_ABNORMAL_BP)

                return anbp_pb2.BPReply(status=ANBP_RES_SUCCEED, bloodPpressure=str(SBP) + '/' + str(DBP) + '/' + str(ppg_ci))

        except Exception as e:
            print(e)
            bp_server_log.logger.error('{} Model calculation error'.format(wear_user_id))
            # 发送消息到钉钉群
            send_ding_message('wear_user_id: {} --> Model calculation error'.format(wear_user_id))
            return anbp_pb2.BPReply(status=ANBP_RES_FAILD)

def server():
    # nn_model = Deeplearningmodel()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    anbp_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:{}'.format(PORT))
    server.start()
    print("服务启动成功...")
    bp_server_log.logger.info("服务启动成功...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:   
        server.stop(0)
        print("服务关闭...")
        bp_server_log.logger.info("服务关闭...")


if __name__ == '__main__':
    server()