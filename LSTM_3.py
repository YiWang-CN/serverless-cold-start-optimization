#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''对单个容器，进行训练和预测，确保模型拟合'''

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

# 定义函数，绘制训练损失和验证损失的图像
def plot_loss(train_loss, val_loss, save_path):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    fig.savefig(save_path)
    plt.close()


# 创建LSTM模型类
class LSTM(nn.Module):
    '''input_size：对应于输入中的特征数量。虽然我们的序列长度是 tw（20），但每个时间不我们只有 1 个值，即到达时刻，因此输入大小将为 1。
hidden_layer_size：指定隐藏层的数量以及每层中神经元的数量。我们将有一层 200 个神经元。
output_size：由于我们要预测未来1个时间步的到达时刻，因此输出大小将为1。'''

    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1,batch_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True) 

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = self.init_hidden(self.batch_size)

        # (num_layers * num_directions, batch_size, hidden_size)
    def init_hidden(self,batch_size):
        # (h_0, c_0)
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_layer_size).to(device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        # LSTM 层期望的输入形状是 (batch_size, seq_len, input_size)/
        # lstm_out 是LSTM层的输出张量，其形状为 ( batch_size, seq_len, hidden_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        h,c = self.hidden_cell
        # h形状转换为(batch_size, hidden_layer_size )
        # predictions 的形状是 (batch_size, output_size)

        predictions = self.linear(h.view(-1, self.hidden_layer_size))

        return predictions


# 训练模型，每20轮的平均损失将被打印
def training(epochs, train_inout_seq, val_inout_seq, model, optimizer,batch_size,model_url):
    
    model = model.to(device)
    model.train() # 设置为训练模式
    epoch_loss = []
    val_loss = []
    for i in range(epochs):
        loss_list = []
        for batch_idx in range(0, int(len(train_inout_seq)/batch_size)):
            batch_data = train_inout_seq[batch_idx:batch_idx+batch_size]
            seq = torch.stack([data[0].to(device) for data in batch_data]).unsqueeze(-1)
            # print(seq.shape)
            labels = torch.stack([data[1].to(device) for data in batch_data])
            # print(labels.shape)

            model.hidden_cell =  model.init_hidden(batch_size)

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels) #标量值
            # print(single_loss.shape)

            optimizer.zero_grad() # 梯度清零，否则梯度会累加
            single_loss.backward() #  是 PyTorch 中张量（Tensor）对象的方法，用于执行反向传播计算梯度
            optimizer.step()
            loss_list.append(single_loss.item())
        mean_loss = sum(loss_list) / len(loss_list)
        epoch_loss.append(mean_loss)

        # 验证集
        val_loss_list = []
        with torch.no_grad():  # 验证集不需要梯度计算
            model.eval()  # 设置为评估模式
            for batch_idx in range(0, int(len(val_inout_seq)/batch_size)):
                batch_data = val_inout_seq[batch_idx:batch_idx+batch_size]
                seq = torch.stack([data[0].to(device) for data in batch_data]).unsqueeze(-1)
                labels = torch.stack([data[1].to(device) for data in batch_data])

                model.hidden_cell = model.init_hidden(batch_size)
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels)
                val_loss_list.append(single_loss.item())
            model.train() # 设置为训练模式
        val_mean_loss = sum(val_loss_list) / len(val_loss_list)
        val_loss.append(val_mean_loss)

        # 每20轮或最后一轮 保存模型参数和画损失函数
        if i%20 == 0 or i == epochs-1:
            save_path = model_url + f'v{i}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # 将损失列表保存到文件中
            with open(save_path + 'epoch_loss.pkl', 'wb') as f:
                pickle.dump(epoch_loss, f)
            with open(save_path + 'val_loss.pkl', 'wb') as f:
                pickle.dump(val_loss, f)

            torch.save(model.state_dict(), save_path + 'model_weights.pth')  # 保存模型参数
            plot_loss(epoch_loss, val_loss, save_path + 'loss.png') #绘制损失函数
            print(f'epoch: {i:3} loss: {mean_loss:10.8f}')

# 定义函数，做预测
def predict(test_input, train_window, model, batch_size=1):
    
    model = model.to(device)
    model.eval() # 设置为评估模式
    predictions = []
    # 测试集转换为张量维度为(1,20,1)，每train_window个为一个输入
    test_input = torch.FloatTensor(test_input).view(-1)
    test_input_seq =[]  # 预测时的输入
    L = len(test_input)
    for i in range(L - train_window):
        test_seq = test_input[i:i + train_window].view(1,train_window,1)
        test_input_seq.append((test_seq))

    with torch.no_grad():  # 是一个PyTorch的上下文管理器，它用于禁用梯度计算
        for seq in test_input_seq:
            seq = seq.to(device)
            model.hidden_cell = model.init_hidden(batch_size)
            predictions.append(model(seq).item())
    y_pred = torch.FloatTensor(predictions)
    labels = test_input[train_window:].view(-1)
    mean_loss = loss_function(y_pred, labels).item()
    model.train() # 设置为训练模式
    return mean_loss, predictions

# 主程序，按流程执行
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    train_window = 20
    dataset_name = 'dataSet_1'
    batch_size = 128
    epochs = 1000
    lr = 0.001

    [requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
    [time_sequences, sequences] = get_time_sequences(requests, metas)
    key = 'roles1'
    # key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
    sequence = sequences[key]
    # 数据做差分
    # sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]


    # 划分训练集：验证集：测试集 = 4:1:1
    total_samples = len(sequence)
    train_samples = int(2/3* total_samples)
    val_samples = int(1/6 * total_samples)

    train_sequence = sequence[:train_samples]
    val_sequence = sequence[train_samples:train_samples+val_samples]
    test_sequence = sequence[train_samples+val_samples:]
    print("数据集划分完成")
    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sequence_normalized = scaler.fit_transform(np.array(train_sequence).reshape(-1, 1))
    val_sequence_normalized = scaler.transform(np.array(val_sequence).reshape(-1, 1))
    test_sequence_normalized = scaler.transform(np.array(test_sequence).reshape(-1, 1))

    # 将 val_sequence_normalized的后20个元素 与 test_sequence_normalized 合并
    test_input = val_sequence_normalized[-train_window:].tolist()
    test_sequence_normalized_list = test_sequence_normalized.tolist()
    test_input.extend(test_sequence_normalized_list)
    test_input = np.array(test_input).reshape(1, -1).tolist()
    test_input = test_input[0]

    # 类型转换为张量，格式转换并打标签
    train_sequence_tensor = torch.FloatTensor(train_sequence_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_sequence_tensor, train_window)
    val_sequence_tensor = torch.FloatTensor(val_sequence_normalized).view(-1)
    val_inout_seq = create_inout_sequences(val_sequence_tensor, train_window)
    
    # # 测试集转换为张量维度为(1,20,1)，每train_window个为一个输入
    # test_input = torch.FloatTensor(test_input).view(-1)
    # test_input_seq =[]  # 预测时的输入
    # L = len(test_input)
    # for i in range(L - train_window):
    #     test_seq = test_input[i:i + train_window].view(1,20,1)
    #     test_input_seq.append((test_seq))

    print("数据预处理完成")

    # 创建LSTM类的对象，定义损失函数和优化器
    model = LSTM(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_function = nn.MSELoss().to(device)
    print("LSTM模型初始化完成，开始训练")

    # 训练模型并保存模型
    model_url = os.path.dirname(os.path.realpath(__file__)) + '/lstm_models/' + dataset_name +'/'+ key +'/'
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    print(key, '容器预测模型开始训练')
    # training(epochs, train_inout_seq, val_inout_seq, model,optimizer,batch_size,model_url)

    # 加载模型
    weights_file = model_url + 'v60/model_weights.pth'
    if os.path.exists(weights_file):
        model = model.to(device)
        model.load_state_dict(torch.load(weights_file,map_location=device))
        print(key, '容器预测模型参数载入完成')
    else:
        print('该模型权重文件不存在')
    
    # 预测
    mean_loss, predictions = predict(test_input, train_window, model, batch_size=1)
    print('mean_loss = ', mean_loss)

    # 归一化值转换为实际值
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    print("预测完成")

    # 预测误差
    error = []
    error_list = []
    mean=0
    error_list = [y - x for x, y in zip(test_sequence,predictions)]
    mean = sum(abs(x) for x in error_list)/len(error_list)
    error.append(mean)
    error.append(error_list)
    # print('error = [mean, error_list]')
    print('error =', error[0])

    # # 将误差列表保存到文件中
    # with open(model_url + 'error_list.pkl', 'wb') as f:
    #     pickle.dump(error_list, f)


    # 记录数据 cold_start_predict,waste_time,exe_time
    # cold_start_predict = []
    # waste_time = []
    # exe_time = []
    # advance_time = 5  # 预热的提前时间(冗余时间)，单位ms




