#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''对容器，进行训练至过拟合
修改隐藏层神经元128个/200个
训练窗口为20/50
添加了warm_start_time，完善了指标统计时预热时间小于前一时刻的错误

对到达时刻做二次差分
按训练批次查看损失曲线
'''
# TODO 批次过大，损失的训练数据过多
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
import csv

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
    plt.xlabel('Batchs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    fig.savefig(save_path)
    plt.close()

# 定义函数，绘制部分训练损失和验证损失的图像
def plot_part_loss(train_loss, val_loss, save_path , start, end):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(start,end),train_loss[start:end], label='Training batch Loss')
    plt.plot(range(start,end),val_loss[start:end], label='Validation Loss')
    plt.xlabel('Batchs')
    plt.ylabel('Loss')
    plt.title(f'{start}-{end} Batchs Training and Validation Loss')
    plt.legend()
    fig.savefig(save_path)
    plt.close()

# 创建LSTM模型类
class LSTM(nn.Module):
    '''input_size：对应于输入中的特征数量。虽然我们的序列长度是 tw（20），但每个时间不我们只有 1 个值，即到达时刻，因此输入大小将为 1。
hidden_layer_size：指定隐藏层的数量以及每层中神经元的数量。我们将有一层 128 个神经元。
output_size：由于我们要预测未来1个时间步的到达时刻，因此输出大小将为1。'''

    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1,batch_size=1):
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


# 训练模型，按批次绘制损失曲线
def training(epochs, train_inout_seq, val_inout_seq, model, optimizer,batch_size,model_url):
    
    model = model.to(device)
    model.train() # 设置为训练模式
    val_loss = []
    loss_list = []
    for i in range(1,epochs+1):

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

            # 验证集
            val_loss_list = []
            with torch.no_grad():  # 验证集不需要梯度计算
                model.eval()  # 设置为评估模式
                for n in range(0, int(len(val_inout_seq)/batch_size)):
                    batch_data = val_inout_seq[n:n+batch_size]
                    seq = torch.stack([data[0].to(device) for data in batch_data]).unsqueeze(-1)
                    labels = torch.stack([data[1].to(device) for data in batch_data])

                    model.hidden_cell = model.init_hidden(batch_size)
                    y_pred = model(seq)
                    single_loss = loss_function(y_pred, labels)
                    val_loss_list.append(single_loss.item())
                model.train() # 设置为训练模式
            val_mean_loss = sum(val_loss_list) / len(val_loss_list)
            val_loss.append(val_mean_loss)

            print(f'epoch:{i:3} batch:{batch_idx+1:3} loss: {single_loss.item():10.8f}; val_loss: {val_mean_loss:10.8f}')
            # 每10批或最后一批次 保存模型参数和画三个损失函数：1~n，5~n，倒数50轮
            if (batch_idx+1)%10 == 0 or batch_idx == int(len(train_inout_seq)/batch_size):
                save_path = model_url + f'v{i}/batch{batch_idx+1}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # 将损失列表保存到文件中
                with open(save_path + 'batch_loss.pkl', 'wb') as f:
                    pickle.dump(loss_list, f)
                with open(save_path + 'val_loss.pkl', 'wb') as f:
                    pickle.dump(val_loss, f)
                torch.save(model.state_dict(), save_path + 'model_weights.pth')  # 保存模型参数
                plot_loss(loss_list, val_loss, save_path + 'loss.png') #绘制0~n损失函数
                if len(loss_list) > 200:
                    plot_part_loss(loss_list, val_loss, save_path + '200-n_loss.png',start=200,end=len(loss_list)) #绘制200~n损失函数
                if len(loss_list) > 400:
                    plot_part_loss(loss_list, val_loss, save_path + 'last_200_loss.png',start=len(loss_list)-200,end=len(loss_list)) #绘制倒数200轮损失函数



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
        device = torch.device("cuda:1")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    train_window = 50
    dataset_name = 'dataSet_2'
    # key = 'roles1'
    key = 'roles2'
    # key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
    batch_size = 128
    epochs = 500
    lr = 0.0001
    epoch_vision = 'v1'
    batch_vision = 'batch400'
    discard = 0
    model_url = os.path.dirname(os.path.realpath(__file__)) + '/lstm_models/' + dataset_name +f'/2diff_cleaned_128_tw={train_window}_'+ key +'/'
    if not os.path.exists(model_url):
        os.makedirs(model_url)

    [requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
    [time_sequences, sequences] = get_time_sequences(requests, metas)
    time_sequences[key] = time_sequences[key][discard:]
    sequences[key] = sequences[key][discard:]
    for ele in metas:
        if ele['key'] == key:
            init_time = ele['initDurationInMs']


    # 数据做差分
    sequence = sequences[key]
    diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
    second_diff_sequence = [diff_sequence[i] - diff_sequence[i-1] for i in range(1, len(diff_sequence))]

    # 划分训练集：验证集：测试集 = 4:1:1
    total_samples = len(second_diff_sequence)
    train_samples = int(0.98 * total_samples)
    val_samples = int(0.01 * total_samples)

    train_sequence = second_diff_sequence[:train_samples]
    val_sequence = second_diff_sequence[train_samples:train_samples+val_samples]
    test_sequence = second_diff_sequence[train_samples+val_samples:]
    print("数据集划分完成")
    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sequence_normalized = scaler.fit_transform(np.array(train_sequence).reshape(-1, 1))
    val_sequence_normalized = scaler.transform(np.array(val_sequence).reshape(-1, 1))
    test_sequence_normalized = scaler.transform(np.array(test_sequence).reshape(-1, 1))

##########################################################
    # # 绘制输入图形
    # input_figure_url = os.path.dirname(os.path.realpath(__file__)) + '/input_figure/' + f'{dataset_name}/'+ key + '/'
    # if not os.path.exists(input_figure_url):
    #     os.makedirs(input_figure_url)

    # # 创建一个包含变量名和值的字典
    # sequence_dict = {
    #     'train_sequence': train_sequence,
    #     'val_sequence': val_sequence,
    #     'test_sequence': test_sequence,
    #     'train_sequence_normalized': train_sequence_normalized,
    #     'val_sequence_normalized': val_sequence_normalized,
    #     'test_sequence_normalized': test_sequence_normalized
    # }
    
    # for sequence_name, sequence_value in sequence_dict.items():  # 输出字典中的变量名和值
    #     input_figure_path = input_figure_url + f'{sequence_name}/'
    #     if not os.path.exists(input_figure_path):
    #         os.makedirs(input_figure_path)

    #     # 预测的时间间隔图像，100个数据一个图形
    #     num_plots = len(sequence_value)
    #     num_plots_per_figure = 300
    #     for i in range(0, num_plots, num_plots_per_figure):
    #         start_index = i
    #         end_index = min(i + num_plots_per_figure, len(sequence_value))
    #         # limit_ymin = 29500
    #         # limit_ymax = 30500
    #         fig = plt.figure(figsize=(15, 10))
    #         plt.plot(range(start_index, end_index), sequence_value[start_index:end_index])
    #         # plt.ylim(ymin=limit_ymin, ymax=limit_ymax)  # 限制y轴范围
    #         plt.xlabel('Arrival order')
    #         plt.ylabel(f'{sequence_name} time interval')
    #         plt.title(key + f'{sequence_name} time interval sequence (Plots {i+1}-{i+num_plots_per_figure})')
    #         fig.savefig(input_figure_path + f"{sequence_name}_time_interval_{i+1}-{i+num_plots_per_figure}.png")
    #         plt.close()
    # print("输入图形绘制完成")
##########################################################
    
    # 将 val_sequence_normalized的后tw个元素 与 test_sequence_normalized 合并
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
    
    print("数据预处理完成")

    # 创建LSTM类的对象，定义损失函数和优化器
    model = LSTM(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_function = nn.MSELoss().to(device)
    print("LSTM模型初始化完成，开始训练")

##########################################################################
# # 绘制输出图像
#     for epoch in range(10,101,10):
#         # 加载模型
#         epoch_vision = f'v{epoch}'
#         weights_file = model_url + epoch_vision + '/model_weights.pth'
#         if os.path.exists(weights_file):
#             model = model.to(device)
#             model.load_state_dict(torch.load(weights_file,map_location=device))
#             print('权重文件为：', weights_file)
#             print(key, '容器预测模型参数载入完成')
#         else:
#             print('该模型权重文件不存在')
#         # 训练集输出
#         train_mean_loss , train_output_normalized = predict(np.array(train_sequence_normalized).reshape(1, -1).tolist(), train_window, model, batch_size=1)
#         train_output = scaler.inverse_transform(np.array(train_output_normalized).reshape(-1, 1)).reshape(-1).tolist()
#         print('train_mean_loss = ', train_mean_loss)
#         # 验证集输出
#         val_mean_loss, val_output_normalized = predict(np.array(val_sequence_normalized).reshape(1, -1).tolist(), train_window, model, batch_size=1)
#         val_output = scaler.inverse_transform(np.array(val_output_normalized).reshape(-1, 1)).reshape(-1).tolist()
#         print('val_mean_loss = ', val_mean_loss)
        
# ########将输入输出的序列写入CSV文件，查看冲击是否能对应上
#         after_50_train_sequence = train_sequence[train_window:]
#         # 确保两个列表长度相同
#         assert len(after_50_train_sequence) == len(train_output), "Length of train_sequence and train_output does not match"
#         csv_url = os.path.dirname(os.path.realpath(__file__)) + '/csv_directory/'+ f'{dataset_name}/'+ key + '/' + f'tw={train_window}_{epoch_vision}' + '/'
#         if not os.path.exists(csv_url):
#             os.makedirs(csv_url)
#         csv_filename = csv_url + f'train_in_out.csv'
#         # 将数据写入CSV文件
#         with open(csv_filename, 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             # 写入标题行
#             writer.writerow(['序号', 'train_input', 'train_output'])
#             # 逐行写入数据
#             for i, (seq, out) in enumerate(zip(after_50_train_sequence, train_output), start=1):
#                 writer.writerow([i, seq, out])

    #     output_figure_url = os.path.dirname(os.path.realpath(__file__)) + '/output_figure/' + f'{dataset_name}/'+ key + '/'+f'train_window={train_window}/' + epoch_vision + '/'
    #     if not os.path.exists(output_figure_url):
    #         os.makedirs(output_figure_url)
        
    #     output_sequence_dict = {
    #         'train_output':train_output,
    #         'train_output_normalized':train_output_normalized,
    #         'val_output':val_output,
    #         'val_output_normalized':val_output_normalized
    #         }
    #     input_sequence_dict = {
    #     'train_sequence': train_sequence,
    #     'train_sequence_normalized': train_sequence_normalized,
    #     'val_sequence': val_sequence,
    #     'val_sequence_normalized': val_sequence_normalized
    #     }

    #     for (output_sequence_name, output_sequence_value), (input_sequence_name, input_sequence_value) in zip(output_sequence_dict.items(), input_sequence_dict.items()):
    #         input_sequence_value = input_sequence_value[50:]
    #         output_figure_path = output_figure_url + f'{output_sequence_name}/'
    #         if not os.path.exists(output_figure_path):
    #             os.makedirs(output_figure_path)
    #         # 预测的时间间隔图像，100个数据一个图形
    #         num_plots = len(output_sequence_value)
    #         num_plots_per_figure = 300
    #         for i in range(0, num_plots, num_plots_per_figure):
    #             start_index = i
    #             end_index = min(i + num_plots_per_figure, len(output_sequence_value))
    #             # limit_ymin = 29500
    #             # limit_ymax = 30500
    #             fig = plt.figure(figsize=(15, 10))
    #             plt.plot(range(start_index, end_index), output_sequence_value[start_index:end_index], label='output sequence', color='red')
    #             plt.plot(range(start_index, end_index), input_sequence_value[start_index:end_index], label='input sequence', color='blue')
    #             # plt.ylim(ymin=limit_ymin, ymax=limit_ymax)  # 限制y轴范围
    #             plt.legend()
    #             plt.xlabel('Arrival order')
    #             plt.ylabel(f'{output_sequence_name} time interval')
    #             plt.title(key + f'{output_sequence_name} time interval sequence (Plots {i+1}-{i+num_plots_per_figure})')
    #             fig.savefig(output_figure_path + f"{output_sequence_name}_time_interval_{i+1}-{i+num_plots_per_figure}.png")
    #             plt.close()
    # print("输出图形绘制完成")
##########################################################################

    # # 训练模型并保存模型
    # print(key, '容器预测模型开始训练')
    # training(epochs, train_inout_seq, val_inout_seq, model,optimizer,batch_size,model_url)




    # 加载模型
    weights_file = model_url + epoch_vision + "/"+batch_vision+'/model_weights.pth'
    if os.path.exists(weights_file):
        model = model.to(device)
        model.load_state_dict(torch.load(weights_file,map_location=device))
        print('权重文件为：', weights_file)
        print(key, '容器预测模型参数载入完成')
    else:
        print('该模型权重文件不存在')
    
    # 预测
    mean_loss, diff_predictions = predict(test_input, train_window, model, batch_size=1)
    print('mean_loss = ', mean_loss)
    # 归一化值转换为实际二次差分值
    second_diff_predictions = scaler.inverse_transform(np.array(diff_predictions).reshape(-1, 1)).reshape(-1).tolist()
    # 根据二次差分，计算时间间隔值
    last_diff_time=diff_sequence[-len(second_diff_predictions)-1:-1]
    diff_predictions = [x + y for x, y in zip(last_diff_time, second_diff_predictions)]
    # 时间间隔值应该>=0，<0的间隔值修正为0
    diff_predictions = [ele if ele >= 0 else 0 for ele in diff_predictions]
    
    # 预测误差
    # error = []
    # error_list = []
    # mean=0
    # error_list = [y - x for x, y in zip(test_sequence,diff_predictions)]
    # mean = sum(abs(x) for x in error_list)/len(error_list)
    # error.append(mean)
    # error.append(error_list)
    # # print('error = [mean, error_list]')
    # print('error =', error[0])
    
    # 根据时间间隔值，计算预测到达时刻
    last_arrival_time=sequence[-len(diff_predictions)-1:-1]
    predictions = [x + y for x, y in zip(last_arrival_time, diff_predictions)]
    # 真实的到达时刻
    actual_value = sequence[-len(predictions):]

    sequence_error_list = [y - x for x, y in zip(actual_value,predictions)]
    sequence_mean_error = sum(abs(x) for x in sequence_error_list)/len(sequence_error_list)
    print('sequence_mean_error =', sequence_mean_error)

    print("预测完成")

# ##########################################################################
# # 绘制预测误差的累积分布图
# sequence_error_int_list = [int(abs(x)) for x in sequence_error_list]
# point_x = 73    ###
# # limit_xmax = float('inf')   ###
# limit_xmin =-153
# limit_xmax =200

# hist, bin_edges = np.histogram(sequence_error_int_list, bins=range(min(sequence_error_int_list), max(sequence_error_int_list) + 1), density=True)
# cumulative_probs = np.cumsum(hist)
# index = np.where(bin_edges >= point_x)[0][0]
# cumulative_prob_at_point = cumulative_probs[index]
# # 绘制累积分布图
# fig = plt.figure(figsize=(15, 10))
# plt.hist(sequence_error_int_list, bins=range(min(sequence_error_int_list), max(sequence_error_int_list) + 1), cumulative=True, density=True, histtype='step', linewidth=1.5)
# plt.xlabel('Absolute value of the forecast error', fontsize=18)
# plt.ylabel('Cumulative Probability',fontsize=18)
# plt.title(key[:5] + f' Container forecast error cumulative distribution(less than {limit_xmax} ms)',fontsize=18)
# plt.xlim(xmin=max(limit_xmin, min(sequence_error_int_list)), xmax=min(limit_xmax, max(sequence_error_int_list)))  # 限制x轴范围
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # # 添加均值mean_error 对应的垂直虚线和水平虚线
# ymin, ymax = plt.gca().get_ylim()
# xmin, xmax = plt.gca().get_xlim()
# # plt.axvline(x=point_x, ymin=0, ymax=cumulative_prob_at_point/ymax, color='gray', linestyle='--')  # 添加垂直虚线
# # plt.axhline(y=cumulative_prob_at_point, xmin= 0,xmax=point_x/(xmax-xmin), color='gray', linestyle='--')  # 添加水平虚线
# # plt.text(point_x, cumulative_prob_at_point, f'({point_x},{cumulative_prob_at_point:.2f})', fontsize=18, color='black', ha='left', va='top')  # 添加对应的坐标
# # plt.scatter(point_x, cumulative_prob_at_point, color='gray', s=20)  # 描绘对应的点
# # 添加水平虚线和对应的垂直虚线
# y_value_max = max(cumulative_probs)
# for y_value in np.arange(0.1,y_value_max, 0.1):
#     x_value = bin_edges[np.where(cumulative_probs >= (y_value))[0][0]]
#     plt.axhline(y=y_value, xmin= 0, xmax=x_value/(xmax-xmin), color='gray', linestyle='--')  # 添加水平虚线
#     plt.axvline(x=x_value, ymin=0, ymax=y_value/ymax, color='gray', linestyle='--')  # 添加对应的垂直虚线
#     plt.text(x_value, y_value, f'({x_value},{y_value:.2f})', fontsize=18, color='black', ha='left', va='top')  # 添加对应的坐标
#     plt.scatter(x_value, y_value, color='gray', s=20)  # 描绘对应的点
# fig.savefig("/home/wangyi/serverless/" + f"predict_error_cumulative_distribution_less_{limit_xmax}_ms.png")
# plt.close()







#########################################################################

    # figure_url = model_url + epoch_vision +'/'
    # fig = plt.figure(figsize=(15, 10))
    # # plt.plot(range(len(sequence)), sequence)
    # # 绘制第一条曲线
    # plt.plot(range(len(sequence)-len(predictions),len(sequence)), sequence[len(sequence)-len(predictions):], label='actual arrival sequence', color='blue')
    # # 绘制第二条曲线
    # plt.plot(range(len(sequence)-len(predictions),len(sequence)), predictions, label='predicted arrival sequence', color='red')
    # plt.legend()
    # plt.xlabel('Arrival order')
    # plt.ylabel('Time of arrival(ms)')
    # plt.title(key + ' container arrival time series')
    # fig.savefig(figure_url + key + "_predicted_arrival_sequence.png")
    # plt.close()

    # fig = plt.figure(figsize=(15, 10))
    # # plt.plot(range(len(sequence)), sequence)
    # # 绘制第一条曲线
    # plt.plot(range(len(diff_sequence)-len(diff_predictions),len(diff_sequence)), diff_sequence[len(diff_sequence)-len(diff_predictions):], label='actual arrival time interval', color='blue')
    # # 绘制第二条曲线
    # plt.plot(range(len(diff_sequence)-len(diff_predictions),len(diff_sequence)), diff_predictions, label='predicted arrival time interval', color='red')
    # plt.legend()
    # plt.xlabel('Arrival order')
    # plt.ylabel('arrival time interval(ms)')
    # plt.title(key + ' Container arrival time interval sequence')
    # fig.savefig(figure_url + key + "_predicted_diff_sequence.png")
    # plt.close()

    # # 预测的时间间隔图像，500个数据一个图形
    # predict_figure_url = os.path.dirname(os.path.realpath(__file__))+'/predict_figure/' +dataset_name+'/' +key+'/' +f'2diff_tw={train_window}_{epoch_vision}_{batch_vision}/'
    # if not os.path.exists(predict_figure_url):
    #     os.makedirs(predict_figure_url)

    # num_plots = len(second_diff_predictions)
    # num_plots_per_figure = 100

    # for i in range(0, num_plots, num_plots_per_figure):
    #     start_index = i
    #     end_index = min(i + num_plots_per_figure, len(second_diff_predictions))
    #     limit_ymin = 29500
    #     limit_ymax = 30500
    #     fig = plt.figure(figsize=(15, 10))
    #     plt.plot(range(start_index, end_index), second_diff_sequence[start_index:end_index], label='second_diff_actually', color='blue')
    #     plt.plot(range(start_index, end_index), second_diff_predictions[start_index:end_index], label='second_diff_predictions', color='red')
    #     plt.legend()
    #     # plt.ylim(ymin=max(limit_ymin,min(diff_predictions[start_index:end_index])), ymax=max(max(diff_predictions[start_index:end_index]),max(diff_sequence[start_index:end_index])))  # 限制y轴范围
    #     # plt.ylim(ymin=limit_ymin, ymax=limit_ymax)  # 限制y轴范围

    #     plt.xlabel('Arrival order')
    #     plt.ylabel('arrival time second diff(ms)')
    #     plt.title(key + f'Container arrival time second diff sequence (Plots {i+1}-{i+num_plots_per_figure})')
    #     fig.savefig(predict_figure_url + key + f"_predicted_diff_sequence_{i+1}-{i+num_plots_per_figure}.png")
    #     plt.close()

    # # 预测的到达时刻图像，500个数据一个图形
    # predict_sequence_figure_url = os.path.dirname(os.path.realpath(__file__))+'/predict_sequence_figure/' +dataset_name+'/' +key+'/' +f'2diff_tw={train_window}_{epoch_vision}_{batch_vision}/'
    # if not os.path.exists(predict_sequence_figure_url):
    #     os.makedirs(predict_sequence_figure_url)

    # num_plots = len(predictions)
    # num_plots_per_figure = 50

    # for i in range(0, num_plots, num_plots_per_figure):
    #     start_index = i
    #     end_index = min(i + num_plots_per_figure, len(predictions))
        
    #     fig = plt.figure(figsize=(15, 10))
    #     plt.plot(range(start_index, end_index), actual_value[start_index:end_index], label='actual arrival time', color='blue')
    #     plt.plot(range(start_index, end_index), predictions[start_index:end_index], label='predicted arrival time', color='red')
    #     plt.legend()
    #     plt.xlabel('Arrival order')
    #     plt.ylabel('arrival time (ms)')
    #     plt.title(key + f'Container arrival sequence (Plots {i+1}-{i+num_plots_per_figure})')
    #     fig.savefig(predict_sequence_figure_url + key + f"_predicted_sequence_{i+1}-{i+num_plots_per_figure}.png")
    #     plt.close()



    # 记录数据 cold_start_predict,waste_time,exe_time
    cold_start_predict={}
    waste_time={}
    exe_time={}
    cold_start_predict[key] = []  # [[start_time,prepare_time],...]
    waste_time[key] = []  # 提前初始化后，等待的时间
    exe_time[key] = []  # [[sys_use_time,request_use_time],...]

    # cold_start_predict={'key':[[start_time,prepare_time],...],...}

    time_sequence = time_sequences[key]
    actual_time_sequence = time_sequence[-len(predictions):]
    # 预热开始的时间
    warm_start_times = [last_arrival_time[i] if predictions[i] - last_arrival_time[i] <init_time else predictions[i] - init_time for i in range(len(predictions))]

    i = 0
    j = 0
    while i < len(actual_time_sequence):
        ele_real = actual_time_sequence[i]
        warm_start_time = warm_start_times[j]
        
        # 判断是否 非热启动
        if ele_real[0] < warm_start_time + init_time:  # 非热启动
            # 判断是否正在预热
            if warm_start_time < ele_real[0]: # 正在预热
                cold_start_predict[key].append([ele_real[0], warm_start_time + init_time - ele_real[0]])
                exe_time[key].append([init_time + ele_real[1], warm_start_time + init_time - ele_real[0] + ele_real[1]])
                # waste_time[key].append()
                i = i + 1
                j = j + 1

            else:  #请求到来时还未开始预热
                cold_start_predict[key].append([ele_real[0], init_time])
                exe_time[key].append([init_time + ele_real[1], init_time + ele_real[1]])
                # waste_time[key].append()
                i = i + 1
                j = j + 1
        else:  # 热启动
            # cold_start_predict[key]
            exe_time[key].append([ele_real[0] - warm_start_time + ele_real[1], ele_real[1]])
            waste_time[key].append(ele_real[0] - warm_start_time - init_time)
            i = i + 1
            j = j + 1

    # 统计指标 cold_statistics,mem_statistics
    cold_statistics = statistics.cold_start_statistics_predict(cold_start_predict, exe_time, metas)
    mem_statistics = statistics.memory_statistics_predict(waste_time, cold_start_predict,exe_time, metas)

    print('cold_statistics:')
    print('cold_statistics[key]=[cold_num, all_num, frequency, cold_time, all_time, utilization, cold_time_every_req]')
    # for key in cold_statistics:
    print(key,":",cold_statistics[key])
    print('mem_statistics:')
    print('mem_statistics[key]=[waste_mem_every_req, waste_mem, req_exe_mem_every_req, req_exe_mem, all_mem, utilization]')
    # for key in mem_statistics:
    print(key,":",mem_statistics[key])
    # print('res_statistics:',res_statistics)
