#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
GPU资源利用太少：加入按批次训练代码√
损失小而误差大：加入dropout层代码
'''
import torch
import torch.nn as nn
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from Platform import input, statistics
from datetime import datetime
import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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


# 创建LSTM模型
class LSTM(nn.Module):
    '''input_size：对应于输入中的特征数量。虽然我们的序列长度是 tw（5），但每个月我们只有 1 个值，即乘客总数，因此输入大小将为 1。
hidden_layer_size：指定隐藏层的数量以及每层中神经元的数量。我们将有一层 200 个神经元。
output_size：输出中的项目数量，由于我们要预测未来1个月的乘客数量，因此输出大小将为1。'''

    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1,batch_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True) 

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = self.init_hidden()

        # (num_layers * num_directions, batch_size, hidden_size)
    def init_hidden(self):
        # (h_0, c_0)
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_layer_size).to(device),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        # LSTM 层期望的输入形状是 (batch_size, seq_len, input_size)/
        # lstm_out 是LSTM层的输出张量，其形状为 ( batch_size, seq_len, hidden_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        h,c = self.hidden_cell
        # h形状转换为(batch_size, hidden_layer_size )
        # predictions 的形状是 (batch_size, output_size)

        predictions = self.linear(h.view(-1, self.hidden_layer_size))

        return predictions


# 训练模型，每轮的平均损失将被打印
def training(epochs, train_inout_seq, model, optimizer,batch_size):
    
    model = model.to(device)
    model.train() # 设置为训练模式
    for i in range(epochs):
        loss_list = []
        # for seq, labels in train_inout_seq:
        #     seq, labels = seq.to(device), labels.to(device)
        for batch_idx in range(0, int(len(train_inout_seq)/batch_size)):
            batch_data = train_inout_seq[batch_idx:batch_idx+batch_size]
            seq = torch.stack([torch.FloatTensor(data[0]).to(device) for data in batch_data]).unsqueeze(-1)
            # print(seq.shape)
            labels = torch.stack([torch.FloatTensor(data[1]).to(device) for data in batch_data])
            # print(labels.shape)

            model.hidden_cell =  model.init_hidden()

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels) #标量值
            # print(single_loss.shape)
            #TODO optimizer为什么只更新一次

            optimizer.zero_grad() # 梯度清零，否则梯度会累加

            single_loss.backward() #  是 PyTorch 中张量（Tensor）对象的方法，用于执行反向传播计算梯度
            optimizer.step()

            loss_list.append(single_loss.item())
        mean_loss = sum(loss_list) / len(loss_list)
        print(f'epoch: {i:3} loss: {mean_loss:10.8f}')

# 做预测
def prediction(test_inputs, fut_pred, train_window, model, test_sequences):

    model.eval() # 设置为评估模式
    predict = []

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        seq = seq.view(1,-1,1).to(device)
        # print(seq.shape)

        with torch.no_grad():  # 是一个PyTorch的上下文管理器，它用于禁用梯度计算
            model.hidden_cell = model.init_hidden()

            test_inputs.append(test_sequences[i])
            predict.append(model(seq).item())
    model.train() # 设置为训练模式
    return predict

# 主程序，按流程执行
if __name__ == "__main__":
    batch_size =1

    # 将输出写入logger文件夹
    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间，生成文件名，例如：2024-02-28_14-30-00.txt
    file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # logger_url = os.path.dirname(os.path.realpath(__file__)) + '/logger/'
    # sys.stdout = Logger(f'{logger_url}{file_name}.log', sys.stdout)
    # sys.stderr = Logger(f'{logger_url}{file_name}.err', sys.stderr)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    train_window = 20

    # 数据集提取
    dataset_name = 'dataSet_1'
    [requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)

    #转换成浮点数

    # 划分训练集和测试集
    test_data_size = int(len(requests) / 2)

    train_data = requests[:-test_data_size]
    test_data = requests[-test_data_size:]

    # 数据集按容器 提取 时间和运行时间 序列 和 时间序列
    [train_data, train_sequences] = get_time_sequences(train_data, metas)
    [test_data, test_sequences] = get_time_sequences(test_data, metas)

    #todo 有未使用的容器
    old_train_sequences = train_sequences.copy()
    for key in old_train_sequences:
        if len(old_train_sequences[key]) <= train_window:
            del train_sequences[key]

    # print(train_sequences)

    old_test_sequences = test_sequences.copy()
    for key in old_test_sequences:
        if len(old_test_sequences[key]) == 0:
            del test_sequences[key]
            del test_data[key]
    print("数据集划分完成")

    # 数据归一化
    scaler = input.createdict(metas)
    train_data_normalized = input.createdict(metas)
    test_data_normalized = input.createdict(metas)
    train_inout_seq = input.createdict(metas)

    for key in train_sequences:
        scaler[key] = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized[key] = scaler[key].fit_transform(
            np.array(train_sequences[key]).reshape(-1, 1))
        test_data_normalized[key] = scaler[key].transform(
            np.array(test_sequences[key]).reshape(-1, 1))

        train_data_normalized[key] = torch.FloatTensor(
            train_data_normalized[key]).view(-1)

        # 格式转换并打标签
        train_inout_seq[key] = create_inout_sequences(
            train_data_normalized[key], train_window)
        # print(key)
        # print(type(train_inout_seq[key][0][0]),train_inout_seq[key][0][0].shape)

    print("数据预处理完成")

    #TODO 优化？？？多个模型并行训练
    # 创建LSTM类的对象，定义损失函数和优化器
    model_dict = input.createdict(metas)
    optimizer_dict = input.createdict(metas)
    # 部分容器没有训练数据，删除对应模型
    keys_ont_in_train_sequence = set(model_dict.keys()) - set(train_sequences.keys())
    for key in keys_ont_in_train_sequence:
        del model_dict[key]
        del optimizer_dict[key]

    loss_function = nn.MSELoss().to(device)
    for key in model_dict:
        model_dict[key] = LSTM(batch_size=batch_size)

        optimizer_dict[key] = torch.optim.Adam(model_dict[key].parameters(),
                                               lr=0.001)
    print("LSTM模型初始化完成，开始训练")

    # 加载模型参数 或者 训练模型并保存模型

    # 检查目录是否存在，不存在则创建
    url = os.path.dirname(os.path.realpath(__file__)) + '/lstm_models/' + dataset_name + '/'
    if not os.path.exists(url):
        os.makedirs(url)


    load_flag = 1

    for key in train_sequences:  #因为有未使用的容器，需要删除键值对，对应的模型不能也不需要训练
        model_name = f'{key}_model_weights.pth'
        path = url + model_name

        if os.path.exists(path) and load_flag:
            model_dict[key] = model_dict[key].to(device)
            model_dict[key].load_state_dict(torch.load(path,
                                            map_location=device))
            print(key, '容器预测模型参数载入完成')

        else:
            print(key, '容器预测模型开始训练')
            epochs = 500
            training(epochs, train_inout_seq[key], model_dict[key],
                     optimizer_dict[key],batch_size)
            torch.save(model_dict[key].state_dict(), path)  # 保存模型参数

    print("模型训练完成，开始预测")

# TODO 单个训练单个预测？
    # 做预测
    predict_dict = input.createdict(metas)
    actual_predictions = input.createdict(metas)
    for key in test_sequences:  #因为有未使用的容器，需要删除键值对，对应的容器不需要预测
        fut_pred = len(test_data_normalized[key])
        test_inputs = train_data_normalized[key][-train_window:].tolist()
        if key not in model_dict:
            pass
        else:
            predict_dict[key] = prediction(test_inputs, fut_pred, train_window,
                                           model_dict[key],
                                           test_data_normalized[key])

        # 归一化值转换为实际值
        actual_predictions[key] = scaler[key].inverse_transform(
            np.array(predict_dict[key]).reshape(-1, 1))
    print("全部预测完成")
    
    # 预测误差
    error = input.createdict(metas)
    all_mean=0
    all_result=[]
    for key in test_sequences:  #因为有未使用的容器，需要删除键值对，对应的容器不需要预测
        result = [y - x for x, y in zip(test_sequences[key],actual_predictions[key])]
        mean = sum(abs(x) for x in result)/len(result)
        error[key].append(mean)
        error[key].append(result)

        all_result.extend(result)
    all_mean = sum(abs(x) for x in all_result)/len(all_result)
    error['all']=[all_mean,all_result]
    print('error[key]=[mean,error_list]')
    print(error)

    # 记录数据 cold_start_predict,waste_time,exe_time
    cold_start_predict = input.createdict(metas)
    waste_time = input.createdict(metas)
    exe_time = input.createdict(metas)
    advance_time = 5  # 预热的提前时间(冗余时间)，单位ms

    # cold_start_predict={'key':[[start_time,prepare_time],...],...}
    for key in test_data:
        init_time = 0
        for ele in metas:
            if ele['key'] == key:
                init_time = ele['initDurationInMs']

        if key not in actual_predictions:  # 没有对应的预测模型的情况，全部冷启动
            for ele in test_data[key]:
                cold_start_predict[key].append([ele[0], init_time])
                exe_time[key].append([ele[0], ele[1] + init_time])
                # waste_time[key]
        else:
            i = 0
            j = 0
            while i < len(test_data[key]):
                ele_real = test_data[key][i]
                ele_predict = actual_predictions[key][j]
                # 判断是否 非热启动
                if ele_real[0] < ele_predict - advance_time:  # 非热启动
                    # 判断是否正在预热
                    if ele_predict - advance_time - init_time < ele_real[0]:
                        cold_start_predict[key].append([
                            ele_real[0],
                            ele_predict - advance_time - ele_real[0]])
                        exe_time[key].append([
                            ele_real[0], ele_predict - advance_time -
                            ele_real[0] + ele_real[1]])
                        # waste_time[key].append()
                        i = i + 1
                        j = j + 1

                    else:  #请求到来时还未开始预热
                        cold_start_predict[key].append([ele_real[0], init_time])
                        exe_time[key].append([ele_real[0],
                                             init_time + ele_real[1]])
                        # waste_time[key].append()
                        i = i + 1
                else:  # 热启动
                    # cold_start_predict[key]
                    exe_time[key].append([ele_real[0], ele_real[1]])
                    waste_time[key].append([ele_real[0] - ele_predict +
                                           advance_time])
                    i = i + 1
                    j = j + 1

    # 统计指标 cold_statistics,mem_statistics
    cold_statistics = statistics.cold_start_statistics_predict(
        cold_start_predict, exe_time, metas)
    mem_statistics = statistics.memory_statistics(waste_time, exe_time, metas)

    print(
        'cold_statistics[key]=[cold_num,all_num,frequency,cold_time,utilization]'
    )
    print(cold_start_predict)
    print('mem_statistics[key]=[waste_mem,all_mem,utilization]')
    print(mem_statistics)
#统计指标  是否问题解决     监督学习指标  网络收敛   准  好
