'''
特殊情况处理
时间间隔分布，按照 秒以内画一张图，剩下的画一张图
'''

from Platform import input
import matplotlib.pyplot as plt
import os

# 定义函数，从数据集里按照容器分类提取 时间序列
def get_sequences(requests, metas):
    '''从数据集里按照容器分类 提取 时间序列
        {key:[starttime,...],...}
    '''
    sequences = input.createdict(metas)

    for ele in requests:
        key = ele["metaKey"]
        sequences[key].append(ele["startTime"])
    # 按先后顺序排序
    for key in sequences:
        sequences[key].sort()
    return sequences

dataset_name = 'dataSet_2'
threshold = 28000
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
sequences = get_sequences(requests, metas)
# key = 'roles1'
key = 'nodes2'
sequence = sequences[key]
# for key in sequences:
#     sequence = sequences[key]
#     if len(sequence) > 10000:
        
figure_url = r"/home/wangyi/serverless/exception_dataset_analysis_detail_figure/" + dataset_name + '/' + f'{key}_figures/'
if not os.path.exists(figure_url):
        os.makedirs(figure_url)

diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
diff_sequence_100s = [x for x in diff_sequence if x < threshold]
diff_sequence_other = [x for x in diff_sequence if x >= threshold]
proportion = len(diff_sequence_100s) / len(diff_sequence)

fig = plt.figure(figsize=(15, 10))
plt.hist(diff_sequence_100s, bins=500)  # 设置500个bins，可以根据实际情况调整
plt.xlabel('arrival time interval(ms)')
plt.ylabel(f'frequency (The proportion is {proportion})')
plt.title(key + ' Container arrival time interval less than 1s distribution')
# plt.grid(True)
fig.savefig(figure_url + key + "_interval_less_1s_distribution.png")
plt.close()


fig = plt.figure(figsize=(15, 10))
plt.hist(diff_sequence_other, bins=500)  # 设置500个bins，可以根据实际情况调整
plt.xlabel('arrival time interval(ms)')
plt.ylabel(f'frequency (The proportion is {1-proportion})')
plt.title(key + ' Container arrival time interval more than 1s distribution')
# plt.grid(True)
fig.savefig(figure_url + key + "_interval_more_1s_distribution.png")
plt.close()