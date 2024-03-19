import matplotlib.pyplot as plt
import os
import pickle
# fig=plt.figure(num=1,figsize=(4,4))
# plt.plot([1,2,3,4],[1,2,3,4])
# plt.show()
# fig.savefig("plot.png")
# a = [i for i in range(10)]
# b = [a[-2:]]
# print(a)
# print(b)

epoch_vision = 'v70'
dataset_name = 'dataSet_2'
key = 'roles2'
model_url = os.path.dirname(os.path.realpath(__file__)) + '/lstm_models/' + dataset_name +'/diff_50_'+ key +'/'
epoch_loss_file = model_url + epoch_vision + '/epoch_loss.pkl'
val_loss_file = model_url + epoch_vision + '/val_loss.pkl'

with open(epoch_loss_file, 'rb') as f:
    epoch_loss = pickle.load(f)
with open(val_loss_file, 'rb') as f:
    val_loss = pickle.load(f)
for i in range(len(epoch_loss)):
    # print('epoch_loss=',epoch_loss)
    # print('val_loss=',val_loss)
    print(f'epoch: {i:3} loss: {epoch_loss[i]:10.8f}  val_loss: {val_loss[i]:10.8f}')


# from Platform import input
# import matplotlib.pyplot as plt
# import os

# '''提取数据集的时序，绘制 到达时间序列，时间间隔分布，时间间隔序列 图像'''

# # 定义函数，从数据集里按照容器分类提取 时间序列
# def get_sequences(requests, metas):
#     '''从数据集里按照容器分类 提取 时间序列
#         {key:[starttime,...],...}
#     '''
#     sequences = input.createdict(metas)

#     for ele in requests:
#         key = ele["metaKey"]
#         sequences[key].append(ele["startTime"])
#     # 按先后顺序排序
#     for key in sequences:
#         sequences[key].sort()
#     return sequences

# dataset_name = 'dataSet_1'
# key = 'roles1'

# [requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
# sequences = get_sequences(requests, metas)

# figure_url = r"/home/wangyi/serverless/dataset_analysis_figure/" + dataset_name + '/'
# if not os.path.exists(figure_url):
#         os.makedirs(figure_url)

# sequence = sequences[key]
# diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

# total_samples = len(diff_sequence)
# train_samples = int(2/3* total_samples)
# val_samples = int(1/6 * total_samples)

# fig = plt.figure(figsize=(15, 10))
# plt.plot(range(len(diff_sequence)), diff_sequence)
# plt.axvline(x=train_samples, color='r', linestyle='--', label='Train Split')
# plt.axvline(x=train_samples + val_samples, color='g', linestyle='--', label='Validation Split')
# plt.xlabel('order')
# plt.ylabel('arrival time interval(ms)')
# plt.title(key + ' Container arrival time interval sequence')
# plt.legend()
# fig.savefig(figure_url + key + "_diff_sequence.png")
# plt.close()

