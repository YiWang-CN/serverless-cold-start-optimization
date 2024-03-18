#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#  先做一个demo 尝试使用lstm √
#  理解功能，按照功能封成函数形式 √
#  理解每一个部件是什么用途，选用更合适的

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from pandas import read_csv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 定义函数，接受原始输入数据，并返回一列元组，输入和标签
def create_inout_sequences(input_data, tw):
    '''该函数将接受原始输入数据，并返回一列元组。
    在每个元组中，第一个元素将包含12项的列表，
    这些项对应于12个月内旅行的乘客数量，
    第二个元组元素将包含一项，即下一个月的乘客数量。
    下一个元组中，12项的数据，其窗口向后移动一位'''
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 创建LSTM模型
class LSTM(nn.Module):
    '''input_size：对应于输入中的特征数量。虽然我们的序列长度是 12，但每个月我们只有 1 个值，即乘客总数，因此输入大小将为 1。
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
def training(epochs, train_inout_seq):
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
def prediction():
    fut_pred = 12

    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    return test_inputs

# 检查GPU是否可用
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available. Please check your installation.")

# 数据集
# print(sns.get_dataset_names())
flight_data = sns.load_dataset("flights")
print(flight_data.head())
print(flight_data.shape)

# 数据可视化
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(flight_data['passengers'])
# plt.show()

# 转换成浮点数
all_data = flight_data['passengers'].values.astype(float)
print(all_data)

# 划分训练集和测试集
test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
# print(len(train_data))
# print(len(test_data))


train_window = 12
# 数据预处理
# def data_preprocessing(train_data,train_window):
# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
# 将数据集转换为张量
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
# print(train_inout_seq)
# return train_data_normalized, train_inout_seq

# train_data_normalized, train_inout_seq = data_preprocessing(train_data,12)

# 创建LSTM类的一个对象，定义丢失函数和优化器
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 150
training(epochs,train_inout_seq)

# 做预测
test_inputs = prediction()

# 归一化值转换为实际值
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)

# 绘图
x = np.arange(132, 144, 1)
print(x)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x,actual_predictions)
plt.show()