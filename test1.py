'''
1）加载数据：构建模型的第一步当然是加载数据集。
2）预处理：根据数据集定义预处理步骤。包括创建时间戳、日期/时间列转换为d类型、序列单变量化等。
3）序列平稳化：为了满足假设，应确保序列平稳。这包括检查序列的平稳性和执行所需的转换。
4）确定d值：为了使序列平稳，执行差分操作的次数将确定为d值。
5）创建ACF和PACF图：这是ARIMA实现中最重要的一步。用ACF PACF图来确定ARIMA模型的输入参数。
6）确定p值和q值：从上一步的ACF和PACF图中读取p和q的值。
7）拟合ARIMA模型：利用我们从前面步骤中计算出来的数据和参数值，拟合ARIMA模型。
8）在验证集上进行预测：预测未来的值。
9）计算RMSE：通过检查RMSE值来检查模型的性能，用验证集上的预测值和实际值检查RMSE值。
'''

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.graphics.api import qqplot
# from statsmodels.tsa.stattools import adfuller
import os
from Platform import input, statistics
import pmdarima as pm

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


dataset_name = 'dataSet_3'
key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
discard = 1500
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
[time_sequences, sequences] = get_time_sequences(requests, metas)
time_sequences[key] = time_sequences[key][discard:]
sequences[key] = sequences[key][discard:]

# 数据做差分
sequence = sequences[key]
# sequence = (np.array(sequence)-sequence[0]).tolist()
diff1_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
diff2_sequence = [diff1_sequence[i] - diff1_sequence[i-1] for i in range(1, len(diff1_sequence))]



# 划分训练集：验证集：测试集 = 4:1:1
total_samples = len(sequence)
train_samples = int(2/3* total_samples)
val_samples = int(1/6 * total_samples)

train_sequence = sequence[:train_samples]
train_diff2 = diff2_sequence[:train_samples]
std_deviation = np.std(train_diff2)

print("标准差:", std_deviation)

val_sequence = sequence[train_samples:train_samples+val_samples]
test_sequence = sequence[train_samples+val_samples:]
print("数据集划分完成")

dta=pd.Series(train_sequence)

model = pm.auto_arima(dta, start_p=0, start_q=0,
                           max_p=1, max_d=5, max_q=1, m=1,
                           start_P=0, seasonal=False,trace=True,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
# model = pm.auto_arima(dta,m=30)
print(model.summary())


# forecast, conf  = model.predict(len(val_sequence), return_conf_int=True)
# forecast = pd.Series(forecast).reset_index(drop=True)
# lower_series = pd.Series(conf[:, 0]).reset_index(drop=True)
# upper_series = pd.Series(conf[:, 1]).reset_index(drop=True)

forecast = model.predict(n_periods=len(val_sequence)).tolist()

# 计算偏差的绝对值
absolute_errors = np.abs(np.subtract(val_sequence, forecast))
# 计算 MAE
mae = np.mean(absolute_errors)
print('MAE:', mae)
fig = plt.figure(figsize=(12, 8))
train_predictions = model.predict_in_sample().tolist()
plt.plot(sequence, label='True')
plt.plot(range(0,len(train_predictions)),train_predictions, label='fit')
plt.plot(range(len(train_predictions),len(train_predictions)+len(forecast)),forecast, label='Predict')
# plt.fill_between(lower_series.index, lower_series, upper_series, 
#                  color='k', alpha=.15)
plt.legend()
plt.title('ARIMA')
plt.savefig(f'./ARIMA_{len(sequence)}.png')
plt.close(fig)






# # 为绘图的连续性把2090的值添加为PredicValue第一个元素
# PredicValue=[]
# PredicValue.append(dta.values[-1])
# for i in range(len(forecast)):
#     PredicValue.append(forecast.iloc[i])
# PredicValue=pd.Series(PredicValue)

# PredicValue.index = pd.Index(sm.tsa.datetools.dates_from_range('2090','2100'))


# fig, ax = plt.subplots(figsize=(12, 8))
# ax = dta.loc['2001':].plot(ax=ax,label='train')
# PredicValue.plot(ax=ax, label='predict')
# plt.legend()
# plt.savefig('ARIMA.png')