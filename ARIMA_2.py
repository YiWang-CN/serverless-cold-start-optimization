from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
import os
from Platform import input, statistics
import pmdarima as pm
# import statistics
from statsmodels.stats.diagnostic import acorr_ljungbox
import pickle
'''
50个数据拟合，预测一个点
'''

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


dataset_name = 'dataSet_2'
# key = 'roles2'
# key = 'nodes1'
key = 'nodes2'
# key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
###############    删除前1500个数据
discard = 0
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
[time_sequences, sequences] = get_time_sequences(requests, metas)
###############

time_sequences[key] = time_sequences[key][discard:]
sequences[key] = sequences[key][discard:]
for ele in metas:
    if ele['key'] == key:
        init_time = ele['initDurationInMs']
# 数据做差分
sequence = sequences[key]
time_sequence = time_sequences[key]
###############  以第一个值为基准，减去第一个值
sequence = (np.array(sequence)-sequence[0]).tolist()
time_sequence =[[sequence[i],time_sequence[i][1]] for i in range(len(time_sequence))]
# diff1_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
# diff2_sequence = [diff1_sequence[i] - diff1_sequence[i-1] for i in range(1, len(diff1_sequence))]



# 划分训练集：验证集：测试集 = 4:1:1
total_samples = len(sequence)
train_samples = int(2/3 * total_samples)
val_samples = int(1/6 * total_samples)

train_sequence = sequence[:train_samples]
val_sequence = sequence[train_samples:train_samples+val_samples]
test_sequence = sequence[train_samples+val_samples:]
print("数据集划分完成")

train_window = 50
p = 0
d = 1
q = 10

figure_url = f'/home/wangyi/serverless/ARIMA_single_step_figures/{key}/({p},{d},{q})/tain_window={train_window}/'
if not os.path.exists(figure_url):
        os.makedirs(figure_url)

model_url = '/home/wangyi/serverless/ARIMA_single_step_model/' + f'{key}/'
if not os.path.exists(model_url):
        os.makedirs(model_url)

# data = diff2_sequence[train_samples-train_window:train_samples+val_samples]
# data = sequence[train_samples-train_window:train_samples+val_samples]  #验证集
 ######################测试集
data = sequence[train_samples+val_samples-train_window:]  
dta=pd.Series(data[:train_window])
dta = dta.reset_index(drop=True)
# dta = train_sequence

model_result = sm.tsa.ARIMA(dta,order=(p,d,q)).fit()
# # model_result = model.fit()
# val_prediction = model_result.forecast(20)
# val_error = np.subtract(val_sequence[:20], val_prediction)
# absolute_errors = np.abs(val_error)
# # 计算 MAE
# mae = np.mean(absolute_errors)
# print('MAE:', mae)
# res = acorr_ljungbox(val_error, lags=15, boxpierce=True, return_df=True)
# print(res)



# 模型在验证集上的预测
# val_model_url = model_url + 'val/'
# if not os.path.exists(val_model_url):
#         os.makedirs(val_model_url)
# val_prediction = []
# for i in range(0, len(val_sequence)):
#     print(i, '/',len(val_sequence)-1)
#     # 预测1个值
#     prediction = model_result.forecast(1)
#     # 将预测值加入结果列表
#     val_prediction.extend(prediction)
#     # # 保存模型
#     # model_result.save(val_model_url + f'{i}.pkl')
#     # 该模型的参数
#     # current_params = model_result.params
#     # 将50个真实值加入dta重新拟合
#     # dta.extend(val_sequence[i:min(i+predict_window, len(val_sequence))])
#     dta = pd.Series(data[i+1:i+train_window])
#     dta = dta.reset_index(drop=True)
#     model_result = sm.tsa.ARIMA(dta,order=(p,d,q)).fit(start_params=model_result.params)
# # model_result.save(val_model_url + f'{len(val_sequence)}.pkl')
# pickle.dump(val_prediction, open(figure_url + 'val_prediction.pkl', 'wb'))
# print(len(val_sequence))
# print(len(val_prediction))
# # val_prediction = pickle.load(open(figure_url + 'val_prediction.pkl', 'rb'))
# val_prediction = val_prediction[:len(val_sequence)]
# # 计算偏差的绝对值偏差的绝对值
# val_error = np.subtract(val_sequence, val_prediction)
# absolute_errors = np.abs(val_error)
# # 计算 MAE
# mae = np.mean(absolute_errors)
# print('MAE:', mae)

# # 白噪声检验
# # 不再指定boxpierce参数，近返回QLB统计量检验结果
# # 同时设置lags参数为一个列表，相应只返回对应延迟阶数的检验结果
# res = acorr_ljungbox(val_error, lags=20, boxpierce=True, return_df=True)
# # res = acorr_ljungbox(data, lags=[6,12,24], return_df=True)
# print(res)

# fig = plt.figure(figsize=(12, 8))
# # train_predictions = model.predict_in_sample().tolist()
# # plt.plot(sequence, label='True')
# # plt.plot(range(0,len(train_predictions)),train_predictions, label='fit')
# # plt.plot(range(len(train_sequence),len(train_sequence)+len(val_prediction)),val_prediction, label='Predict')
# plt.plot(val_sequence, label='True')
# plt.plot(val_prediction, label='Predict')
# plt.legend()
# plt.savefig(figure_url + key + '_ARIMA.png')
# plt.close(fig)

# fig = plt.figure(figsize=(12, 8))
# plt.plot(val_sequence[:100], label='True')
# plt.plot(val_prediction[:100], label='Predict')
# plt.legend()
# plt.savefig(figure_url + key + '_ARIMA_100.png')
# plt.close(fig)



# # 模型在测试集上的预测
test_figure_url = figure_url + 'test/'
if not os.path.exists(test_figure_url):
        os.makedirs(test_figure_url)
test_prediction = []
for i in range(0, len(test_sequence)):
    print(i, '/',len(test_sequence)-1)
    # 预测1个值
    prediction = model_result.forecast(1)
    # 将预测值加入结果列表
    test_prediction.extend(prediction)
    # # 保存模型
    # model_result.save(val_model_url + f'{i}.pkl')
    # 该模型的参数
    # current_params = model_result.params
    # 将50个真实值加入dta重新拟合
    # dta.extend(val_sequence[i:min(i+predict_window, len(val_sequence))])
    dta = pd.Series(data[i+1:i+train_window])
    dta = dta.reset_index(drop=True)
    model_result = sm.tsa.ARIMA(dta,order=(p,d,q)).fit(start_params=model_result.params)
# model_result.save(val_model_url + f'{len(val_sequence)}.pkl')
pickle.dump(test_prediction, open(test_figure_url + 'test_prediction.pkl', 'wb'))
print(len(test_sequence))
print(len(test_prediction))
# val_prediction = pickle.load(open(figure_url + 'val_prediction.pkl', 'rb'))
test_prediction = test_prediction[:len(test_sequence)]
# 计算偏差的绝对值偏差的绝对值
test_error = np.subtract(test_sequence, test_prediction)
absolute_errors = np.abs(test_error)
# 计算 MAE
mae = np.mean(absolute_errors)
print('MAE:', mae)

# 白噪声检验
# 不再指定boxpierce参数，近返回QLB统计量检验结果
# 同时设置lags参数为一个列表，相应只返回对应延迟阶数的检验结果
res = acorr_ljungbox(test_error, lags=20, boxpierce=True, return_df=True)
# res = acorr_ljungbox(data, lags=[6,12,24], return_df=True)
print(res)

fig = plt.figure(figsize=(12, 8))
# train_predictions = model.predict_in_sample().tolist()
# plt.plot(sequence, label='True')
# plt.plot(range(0,len(train_predictions)),train_predictions, label='fit')
# plt.plot(range(len(train_sequence),len(train_sequence)+len(val_prediction)),val_prediction, label='Predict')
plt.plot(test_sequence, label='True')
plt.plot(test_prediction, label='Predict')
plt.legend()
plt.savefig(test_figure_url + key + '_ARIMA.png')
plt.close(fig)

fig = plt.figure(figsize=(12, 8))
plt.plot(test_sequence[:100], label='True')
plt.plot(test_prediction[:100], label='Predict')
plt.legend()
plt.savefig(test_figure_url + key + '_ARIMA_100.png')
plt.close(fig)


test_prediction = pickle.load(open(test_figure_url + 'test_prediction.pkl', 'rb'))
test_prediction = test_prediction[:len(test_sequence)]
# 记录数据 cold_start_predict,waste_time,exe_time
cold_start_predict={}
waste_time={}
exe_time={}
cold_start_predict[key] = []  # [[start_time,prepare_time],...]
waste_time[key] = []  # 提前初始化后，等待的时间
exe_time[key] = []  # [[sys_use_time,request_use_time],...]

# cold_start_predict={'key':[[start_time,prepare_time],...],...}
predictions = test_prediction
last_arrival_time=sequence[-len(predictions)-1:-1]
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

# print('cold_statistics[key]=[cold_num, all_num, frequency, cold_time, all_time, utilization, cold_time_every_req]')
# print(cold_statistics)
# print('mem_statistics[key]=[waste_mem_every_req, waste_mem, req_exe_mem_every_req, req_exe_mem, all_mem, utilization]')
# print(mem_statistics)
print('cold_statistics:')
print('cold_statistics[key]=[cold_num, all_num, frequency, cold_time, all_time, utilization, cold_time_every_req]')
# for key in cold_statistics:
print(key,":",cold_statistics[key])
print('mem_statistics:')
print('mem_statistics[key]=[waste_mem_every_req, waste_mem, req_exe_mem_every_req, req_exe_mem, all_mem, utilization]')
# for key in mem_statistics:
print(key,":",mem_statistics[key])
# print('res_statistics:',res_statistics)