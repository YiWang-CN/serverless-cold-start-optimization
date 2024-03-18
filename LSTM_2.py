#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Platform import input,statistics
'''


'''
# 定义函数，从数据集里按照容器分类提取 时间和运行时间 序列 和 时间序列
def get_time_sequences(requests,metas):
    '''从数据集里按照容器分类 提取 时间和运行时间 序列 和 时间序列'''
    time_sequences = input.createdict(metas)
    sequences = input.createdict(metas)

    for ele in requests :
        key = ele["metaKey"]
        time_sequences[key].append([ele["startTime"],ele["durationsInMs"]])
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
    return [time_sequences,sequences]

# 定义函数，接受原始输入数据，并返回一列元组，输入和标签
def create_inout_sequences(input_data, tw):
    '''该函数将接受原始输入数据，并返回一列元组。
    在每个元组中，有tw个数据（输入）和下一个的数值（标签），'''
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 创建LSTM模型
class LSTM(nn.Module):
    '''input_size：对应于输入中的特征数量。虽然我们的序列长度是 tw（5），但每个月我们只有 1 个值，即乘客总数，因此输入大小将为 1。
hidden_layer_size：指定隐藏层的数量以及每层中神经元的数量。我们将有一层 200 个神经元。
output_size：输出中的项目数量，由于我们要预测未来1个月的乘客数量，因此输出大小将为1。'''
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 训练模型，每隔25个迭代，损失将被打印
def training(epochs,train_inout_seq,model,optimizer):
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 做预测
def prediction(test_inputs, fut_pred, train_window, model, test_sequences):

    model.eval()
    predict=[]

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():    # 是一个PyTorch的上下文管理器（context manager），它用于禁用梯度计算。
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            # todo 每次添加真实值作为新输入值  预测值？
            test_inputs.append(test_sequences[i])
            predict.append(model(seq).item())
    return predict



# 主程序，按流程执行
if __name__=="__main__":
    # todo train_window大小
    train_window = 20

    # 数据集提取
    [requests,metas] = input.input(r"F:\python_file\serverless\dataSet_1")

    #转换成浮点数


    # 划分训练集和测试集
    test_data_size = int(len(requests)/2)

    train_data = requests[:-test_data_size]
    test_data = requests[-test_data_size:]

    # 数据集按容器 提取 时间和运行时间 序列 和 时间序列
    [train_data, train_sequences] = get_time_sequences(train_data, metas)
    [test_data, test_sequences] = get_time_sequences(test_data, metas)

    #todo 有未使用的容器
    old_train_sequences=train_sequences.copy()
    for key in old_train_sequences:
        if len(old_train_sequences[key]) <= train_window:
            del train_sequences[key]

    # print(train_sequences)

    old_test_sequences = test_sequences.copy()
    for key in old_test_sequences:
        if len(old_test_sequences[key]) == 0:
            del test_sequences[key]
    print("数据集划分完成")

    # 数据归一化
    scaler = input.createdict(metas)
    train_data_normalized = input.createdict(metas)
    test_data_normalized = input.createdict(metas)
    train_inout_seq = input.createdict(metas)

    for key in train_sequences:
        scaler[key] = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized[key] = scaler[key].fit_transform(np.array(train_sequences[key]).reshape(-1, 1))
        test_data_normalized[key] = scaler[key].transform(np.array(train_sequences[key]).reshape(-1, 1))

        train_data_normalized[key] = torch.FloatTensor(train_data_normalized[key]).view(-1)

        # 格式转换并打标签
        train_inout_seq[key] = create_inout_sequences(train_data_normalized[key], train_window)
    print("数据预处理完成")

    #todo 优化？？？
    # 创建LSTM类的对象，定义丢失函数和优化器
    model_dict = input.createdict(metas)
    optimizer_dict = input.createdict(metas)
    loss_function = nn.MSELoss()
    for key in model_dict:
        model_dict[key]=LSTM()

        optimizer_dict[key] = torch.optim.Adam(model_dict[key].parameters(), lr=0.001)
    print("LSTM模型初始化完成，开始训练")

    # 训练模型
    for key in train_sequences: #因为有未使用的容器，需要删除键值对，对应的模型不能也不需要训练
        print(key,'容器预测模型开始训练')
        epochs = 150
        training(epochs, train_inout_seq[key], model_dict[key], optimizer_dict[key])

    print("模型训练完成，开始预测")

    # 做预测
    predict_dict = input.createdict(metas)
    actual_predictions = input.createdict(metas)
    for key in test_sequences: #因为有未使用的容器，需要删除键值对，对应的容器不需要预测
        fut_pred = len(test_data_normalized[key])
        test_inputs = train_data_normalized[key][-train_window:].tolist()
        if key not in model_dict:
            #todo
            pass
        else:
            predict_dict[key] = prediction(test_inputs, fut_pred, train_window, model_dict[key], test_data_normalized[key])

        # 归一化值转换为实际值
        actual_predictions[key] = scaler[key].inverse_transform(np.array(predict_dict[key]).reshape(-1, 1))
    print("全部预测完成")
    # 记录数据 cold_start,waste_time,exe_time

    # 统计指标 cold_statistics,mem_statistics


#统计指标  是否问题解决     监督学习指标  网络收敛   准  好