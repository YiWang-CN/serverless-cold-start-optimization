from Platform import input
import matplotlib.pyplot as plt
import os

'''提取数据集的时序，绘制 到达时间序列，时间间隔分布，时间间隔序列 图像'''

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


dataset_name = 'dataSet_3'
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
sequences = get_sequences(requests, metas)

figure_url = r"/home/wangyi/serverless/dataset_analysis_figure/" + dataset_name + '/'
if not os.path.exists(figure_url):
        os.makedirs(figure_url)

for key in sequences:
    sequence = sequences[key]
    if len(sequence) > 10000:
        # key = 'nodes1'
        # sequence = sequences['nodes1']
        diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

        fig = plt.figure(figsize=(15, 10))
        plt.plot(range(len(sequence)), sequence)
        plt.xlabel('Arrival order')
        plt.ylabel('Time of arrival(ms)')
        plt.title(key + ' container arrival time series')
        # total_data_number = len(sequence)
        # plt.text(1, 0.5, f'Total number of container requests: {total_data_number}', fontsize=20, color='red')
        fig.savefig(figure_url + key + "_arrival_sequence.png")
        

        fig = plt.figure(figsize=(15, 10))
        plt.hist(diff_sequence, bins=30)  # 设置30个bins，可以根据实际情况调整
        plt.xlabel('arrival time interval(ms)')
        plt.ylabel('frequency')
        plt.title(key + ' Container arrival time interval distribution')
        # plt.grid(True)
        fig.savefig(figure_url + key + "_diff_distribution.png")
        plt.close()

        fig = plt.figure(figsize=(15, 10))
        plt.plot(range(len(diff_sequence)), diff_sequence)
        plt.xlabel('order')
        plt.ylabel('arrival time interval(ms)')
        plt.title(key + ' Container arrival time interval sequence')
        fig.savefig(figure_url + key + "_diff_sequence.png")
        plt.close()
