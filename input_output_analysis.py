'''输入    训练  验证  测试
真实
归一

输出      训练    验证
真实
归一
'''
import torch
import torch.nn as nn
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from Platform import input, statistics
from datetime import datetime
import sys
import pickle

# 定义函数，从数据集里按照容器分类提取 时间和运行时间 序列 和 时间序列
def get_time_sequences(requests, metas):
    '''从数据集里按照容器分类 提取 时间和运行时间 序列 和 时间序列
    [
        {key:[[starttime,durationsinms],...],...},
        {key:[starttime,...],...}
    ]
    '''
    time_sequences = input.createdict(metas)
    sequences = input.createdict(metas)

    for ele in requests:
        key = ele["metaKey"]
        time_sequences[key].append([ele["startTime"], ele["durationsInMs"]])
        sequences[key].append(ele["startTime"])

    # 按先后顺序排序
    # 获取列表的第1个元素
    def takefirst(elem):
        return elem[0]

    # 指定按第一个元素排序
    for key in time_sequences:
        time_sequences[key].sort(key=takefirst)

    for key in sequences:
        sequences[key].sort()
    return [time_sequences, sequences]

train_window = 50
dataset_name = 'dataSet_1'
key = 'nodes1'
batch_size = 128
epochs = 5000
lr = 0.0001
epoch_vision = 'v2600'
discard = 0
model_url = os.path.dirname(os.path.realpath(__file__)) + '/lstm_models/' + dataset_name +f'/diff_cleaned_128_tw={train_window}_'+ key +'/'
# if not os.path.exists(model_url):
#     os.makedirs(model_url)

[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
[time_sequences, sequences] = get_time_sequences(requests, metas)
time_sequences[key] = time_sequences[key][discard:]
sequences[key] = sequences[key][discard:]
for ele in metas:
    if ele['key'] == key:
        init_time = ele['initDurationInMs']


