import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from Platform import input, statistics
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
'''
时序数据 ADF 检验、KPSS 检验、白噪声检验、ACF和PACF
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
# key = 'rolebindings2'
key = 'nodes2'
# key = 'roles2'
# key = 'nodes1'
# key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
figure_path = "/home/wangyi/serverless/acf_pacf_figures/"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
[time_sequences, sequences] = get_time_sequences(requests, metas)
for ele in metas:
    if ele['key'] == key:
        init_time = ele['initDurationInMs']

# 数据做差分
#############################################
# discard = int(len(sequences[key])*5/6)
discard = 0
sequence = sequences[key][discard:]
diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
second_diff_sequence = [diff_sequence[i] - diff_sequence[i-1] for i in range(1, len(diff_sequence))]

# 划分训练集：验证集：测试集 = 4:1:1
total_samples = len(sequence)
train_samples = int(2/3* total_samples)
val_samples = int(1/6 * total_samples)

train_sequence = sequence[:train_samples]
# train_diff2_sequence = second_diff_sequence[:train_samples]

val_sequence = sequence[train_samples:train_samples+val_samples]
test_sequence = sequence[train_samples+val_samples:]
print("数据集划分完成")

data = diff_sequence

# ADF 检验
adf_result = adfuller(data)
print("ADF Test:")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

# KPSS 检验
kpss_result = kpss(data)
print("\nKPSS Test:")
print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])
print("Critical Values:", kpss_result[3])

# 白噪声检验
# 不再指定boxpierce参数，近返回QLB统计量检验结果
# 同时设置lags参数为一个列表，相应只返回对应延迟阶数的检验结果
res = acorr_ljungbox(data, lags=[6,12,24], boxpierce=True, return_df=True)
# res = acorr_ljungbox(data, lags=[6,12,24], return_df=True)
print(res)


data_array = np.array(data)
# 绘制ACF和PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
# 绘制 ACF（自相关函数）图
plot_acf(data_array, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
# 绘制 PACF（偏自相关函数）图
plot_pacf(data_array, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

# plt.tight_layout()

# plot_acf(data, ax=ax, lags=50, title='Autocorrelation Function (ACF)')
# plot_pacf(data, ax=ax, lags=50, title='Partial Autocorrelation Function (PACF)')
plt.savefig(figure_path + key + "_acf_pacf.png")