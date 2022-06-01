#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append(r'nn_models')

import numpy as np
from os import walk
from os.path import join
import pandas as pd
from keras.models import Sequential, model_from_json
import json
import os
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import time
import keras
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from matplotlib import pyplot
from scipy import stats
from keras.models import load_model
import h5py



class Deeplearningmodel():
    def __init__(self):
        self.model_common = load_model(r'./nn_models/models/common.h5')
        # self.model_special = load_model(r'./nn_models/models/special.h5')

    '''
    def model_predict(self,data):
        """
        深度学习模型预测血压
        :param data:
        :return:ndarray,一维两个的高压和低压值
        """
        result = self.model_special.predict(data)
        if result[0] <= 93 or result[1] >= 138:
            return result
        else:
            result = self.model_common.predict(data)
            return result
    '''


    def model_predict(self,data):
        """
        深度学习模型预测血压
        :param data:
        :return:ndarray,一维两个的高压和低压值
        """
        result = self.model_common.predict(data)
        return result


if __name__ == '__main__':

    # 异常值测试

    # data = pd.read_csv(r'E:\project\ANBP\code\branches\jianbin.xu\python\my_own_train\personal_models_get_features\data_0825_result\5ae46115_ppg_data.csv')
    # x = data[:][['A1/(A1+A2)','A2/(A1+A2)','A1/AC','A2/AC','Slope',
    #              'DiastolicTime','SystolicTime','RR','Age','Gender','Height','Weight','UpToMaxSlopeTime']]
    # X_test = x.values
    model = Deeplearningmodel()
    result = model.model_predict()










