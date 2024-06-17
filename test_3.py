import pickle
import numpy as np
from Platform import input, statistics
import matplotlib.pyplot as plt



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
###############    删除前1500个数据
discard = 1500
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
[time_sequences, sequences] = get_time_sequences(requests, metas)
time_sequences[key] = time_sequences[key][discard:]
sequences[key] = sequences[key][discard:]

# 数据做差分
sequence = sequences[key]
################  以第一个值为基准，减去第一个值
sequence = (np.array(sequence)-sequence[0]).tolist()
diff1_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
diff2_sequence = [diff1_sequence[i] - diff1_sequence[i-1] for i in range(1, len(diff1_sequence))]



# 划分训练集：验证集：测试集 = 4:1:1
total_samples = len(sequence)
train_samples = int(2/3* total_samples)
val_samples = int(1/6 * total_samples)

train_sequence = sequence[:train_samples]
val_sequence = sequence[train_samples:train_samples+val_samples]
test_sequence = sequence[train_samples+val_samples:]
print("数据集划分完成")

train_window = 50
p = 0
d = 0
q = 1

figure_url = f'/home/wangyi/serverless/ARIMA_single_step_figures/{key}/({p},{d},{q})/tain_window={train_window}/'

val_prediction = pickle.load(open(figure_url + 'val_prediction.pkl', 'rb'))

val_prediction = val_prediction[:len(val_sequence)]


# data = diff2_sequence[train_samples-train_window:train_samples+val_samples]
# 还原二次差分
last_diff_time=diff1_sequence[train_samples:train_samples+val_samples]
diff_predictions = [x + y for x, y in zip(last_diff_time, val_prediction)]

last_arrival_time=sequence[train_samples+1:train_samples+1+val_samples]
val_prediction = [x + y for x, y in zip(last_arrival_time, diff_predictions)]

# 计算偏差的绝对值偏差的绝对值
val_error = np.subtract(val_sequence, val_prediction)
absolute_errors = np.abs(val_error)
# 计算 MAE
mae = np.mean(absolute_errors)
print('MAE:', mae)


fig = plt.figure(figsize=(12, 8))
plt.plot(val_sequence, label='True')
plt.plot(val_prediction, label='Predict')
plt.legend()
plt.savefig(figure_url + key + '_ARIMA.png')
plt.close(fig)

fig = plt.figure(figsize=(12, 8))
plt.plot(val_sequence[:100], label='True')
plt.plot(val_prediction[:100], label='Predict')
plt.legend()
plt.savefig(figure_url + key + '_ARIMA_100.png')
plt.close(fig)
