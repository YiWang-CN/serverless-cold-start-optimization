'''到达时刻的时序图

到达间隔的时序图
到达间隔的分布图
到达间隔的累积分布图

运行时间的时序图
运行时间的分布图
运行时间的累积分布图'''

from Platform import input
import matplotlib.pyplot as plt
import os
import numpy as np

# 定义函数，从数据集里按照容器分类提取 时间序列
def get_sequences(requests, metas):
    '''从数据集里按照容器分类 提取 时间序列
        {key:[starttime,...],...}
    '''
    sequences = input.createdict(metas)
    durations = input.createdict(metas)

    for ele in requests:
        key = ele["metaKey"]
        sequences[key].append(ele["startTime"])
        durations[key].append(ele["durationsInMs"])
    # 按先后顺序排序
    for key in sequences:
        sequences[key].sort()
    return sequences,durations

dataset_name = 'dataSet_1'
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
sequences,durations = get_sequences(requests, metas)
key = 'nodes1'

sequence = sequences[key]
duration = durations[key]
diff_sequence = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
for ele in metas:
    if ele['key'] == key:
        init_time = ele['initDurationInMs']

figure_url = r"/home/wangyi/serverless/containers_analysis_figure/" + dataset_name + '/' +f'container_{key}/'
if not os.path.exists(figure_url):
        os.makedirs(figure_url)

# 到达间隔的时序图, 使用 diff_interval_threshold，查看大于阈值的细节
diff_interval_threshold = 29900   ###
# dotted_line_y = 5000    ###
diff_sequence_mean = np.mean(diff_sequence)
diff_sequence_variance = np.var(diff_sequence)
fig = plt.figure(figsize=(15, 10))
# 布尔索引，选择y轴大于diff_interval_threshold的部分
selected_diff_sequence = [diff for diff in diff_sequence if diff > diff_interval_threshold]
selected_indices = [i for i, diff in enumerate(diff_sequence) if diff > diff_interval_threshold]
plt.plot(selected_indices, selected_diff_sequence)
# plt.axhline(y = dotted_line_y, color='r', linestyle='--')
plt.xlabel('order', fontsize=18)
plt.ylabel('arrival time interval(ms)',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval sequence (more than {diff_interval_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {diff_sequence_mean:<.2f}\n{variance_label:<{max_label_length}} {diff_sequence_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url + f"diff_sequence_more_{diff_interval_threshold}_ms.png")
plt.close()

# 到达间隔的累积分布图,使用threshold，查看大于阈值的细节
# 确定 init_time 和 mean_run_time 对应的累积概率
limit_xmin= 29000    ###
hist, bin_edges = np.histogram(diff_sequence, bins=range(min(diff_sequence), max(diff_sequence) + 1), density=True)
cumulative_probs = np.cumsum(hist)
init_time_index = np.where(bin_edges >= init_time)[0][0]
cumulative_prob_at_init_time = cumulative_probs[init_time_index] # init_time 对应的累积概率
mean_run_time = np.mean(duration)
mean_run_time_index = np.where(bin_edges >= mean_run_time)[0][0]
cumulative_prob_at_mean_run_time = cumulative_probs[mean_run_time_index] # mean_run_time 对应的累积概率
# 绘制累积分布图
fig = plt.figure(figsize=(15, 10))
plt.hist(diff_sequence, bins=range(min(diff_sequence), max(diff_sequence) + 1), cumulative=True, density=True, histtype='step', linewidth=1.5)
plt.xlabel('arrival time interval(ms)',fontsize=18)
plt.ylabel('Cumulative Probability',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval cumulative distribution(more than {limit_xmin}ms)',fontsize=18)
plt.xlim(xmin=max(limit_xmin,min(diff_sequence)), xmax=max(diff_sequence))  # 限制x轴范围
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 添加init_time对应的垂直虚线和水平虚线
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
plt.axvline(x=init_time, ymin=0, ymax=cumulative_prob_at_init_time/ymax, color='r', linestyle='--')  # 添加垂直虚线
plt.axhline(y=cumulative_prob_at_init_time, xmin= 0,xmax=init_time/xmax, color='r', linestyle='--')  # 添加水平虚线
plt.text(init_time, cumulative_prob_at_init_time, f'({init_time},{cumulative_prob_at_init_time:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
plt.text(init_time, -0.01, f'init_time:{init_time}', fontsize=15, color='black', ha='center', va='top')  # 添加对应的x轴坐标
plt.text(0, cumulative_prob_at_init_time, f'{cumulative_prob_at_init_time:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.scatter(init_time, cumulative_prob_at_init_time,color='r', s=20)  # 描绘对应的点
# 添加mean_run_time对应的垂直虚线和水平虚线
plt.axvline(x=mean_run_time, ymin=0, ymax=cumulative_prob_at_mean_run_time/ymax, color='g', linestyle='--')  # 添加垂直虚线
plt.axhline(y=cumulative_prob_at_mean_run_time, xmin= 0,xmax=mean_run_time/xmax, color='g', linestyle='--')  # 添加水平虚线
plt.text(mean_run_time, cumulative_prob_at_mean_run_time, f'({mean_run_time:.2f},{cumulative_prob_at_mean_run_time:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
plt.text(mean_run_time, 0.04, f'mean_run_time:{mean_run_time:.2f}', fontsize=15, color='black', ha='center', va='top')  # 添加对应的x轴坐标
plt.text(0, cumulative_prob_at_mean_run_time, f'{cumulative_prob_at_mean_run_time:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.scatter(mean_run_time, cumulative_prob_at_mean_run_time, color='g', s=20)  # 描绘对应的点
# 每10%描出坐标，添加水平虚线和对应的垂直虚线
plt.axhline(y=1, color='gray', linestyle='--')  # 添加水平虚线
y_value_max = max(cumulative_probs)
for y_value in np.arange(0.1,y_value_max, 0.1):
    x_value = bin_edges[np.where(cumulative_probs >= (y_value))[0][0]]
    plt.axhline(y=y_value, xmin= 0, xmax=x_value/xmax, color='gray', linestyle='--')  # 添加水平虚线
    plt.axvline(x=x_value, ymin=0, ymax=y_value/ymax, color='gray', linestyle='--')  # 添加对应的垂直虚线
    plt.text(x_value, y_value, f'({x_value},{y_value:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
    plt.scatter(x_value, y_value, color='gray', s=20)  # 描绘对应的点
fig.savefig(figure_url + f"diff_cumulative_distribution_more_{limit_xmin}_ms.png")
plt.close()

# 到达间隔的时序图, 使用 diff_interval_threshold，查看细节
diff_interval_threshold = 10000   ###
dotted_line_y = 5000    ###
diff_sequence_mean = np.mean(diff_sequence)
diff_sequence_variance = np.var(diff_sequence)
fig = plt.figure(figsize=(15, 10))
# 布尔索引，选择y轴小于diff_interval_threshold的部分
selected_diff_sequence = [diff for diff in diff_sequence if diff < diff_interval_threshold]
selected_indices = [i for i, diff in enumerate(diff_sequence) if diff < diff_interval_threshold]
plt.plot(selected_indices, selected_diff_sequence)
plt.axhline(y = dotted_line_y, color='r', linestyle='--')
plt.xlabel('order', fontsize=18)
plt.ylabel('arrival time interval(ms)',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval sequence (less than {diff_interval_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {diff_sequence_mean:<.2f}\n{variance_label:<{max_label_length}} {diff_sequence_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url + f"diff_sequence_less_{diff_interval_threshold}_ms.png")
plt.close()
# 到达间隔的时序图, 使用 diff_interval_threshold，查看细节
diff_interval_threshold = 100   ###
dotted_line_y = 50    ###
diff_sequence_mean = np.mean(diff_sequence)
diff_sequence_variance = np.var(diff_sequence)
fig = plt.figure(figsize=(15, 10))
# 布尔索引，选择y轴小于diff_interval_threshold的部分
selected_diff_sequence = [diff for diff in diff_sequence if diff < diff_interval_threshold]
selected_indices = [i for i, diff in enumerate(diff_sequence) if diff < diff_interval_threshold]
plt.plot(selected_indices, selected_diff_sequence)
plt.axhline(y = dotted_line_y, color='r', linestyle='--')
plt.xlabel('order', fontsize=18)
plt.ylabel('arrival time interval(ms)',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval sequence (less than {diff_interval_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {diff_sequence_mean:<.2f}\n{variance_label:<{max_label_length}} {diff_sequence_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url + f"diff_sequence_less_{diff_interval_threshold}_ms.png")
plt.close()
# 到达间隔的时序图, 使用 diff_interval_threshold，查看细节
diff_interval_threshold = 2000   ###
dotted_line_y = 1000    ###
diff_sequence_mean = np.mean(diff_sequence)
diff_sequence_variance = np.var(diff_sequence)
fig = plt.figure(figsize=(15, 10))
# 布尔索引，选择y轴小于diff_interval_threshold的部分
selected_diff_sequence = [diff for diff in diff_sequence if diff < diff_interval_threshold]
selected_indices = [i for i, diff in enumerate(diff_sequence) if diff < diff_interval_threshold]
plt.plot(selected_indices, selected_diff_sequence)
plt.axhline(y = dotted_line_y, color='r', linestyle='--')
plt.xlabel('order', fontsize=18)
plt.ylabel('arrival time interval(ms)',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval sequence (less than {diff_interval_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {diff_sequence_mean:<.2f}\n{variance_label:<{max_label_length}} {diff_sequence_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url + f"diff_sequence_less_{diff_interval_threshold}_ms.png")
plt.close()

# 运行时间的时序图, 使用 duration_threshold，查看细节
duration_threshold = 100   ###
selected_duration_sequence = [ele for ele in duration if ele < duration_threshold]
selected_duration_indices = [i for i, ele in enumerate(duration) if ele < duration_threshold]
duration_mean = np.mean(duration)
duration_variance = np.var(duration)
fig = plt.figure(figsize=(15, 10))
plt.plot(selected_duration_indices, selected_duration_sequence)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
plt.axhline(y=duration_mean, color='r', linestyle='--') # 添加均值线
plt.text(-0.08*xmax, duration_mean, f'{duration_mean:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.xlabel('Arrival order', fontsize=18)  # 调整x轴标签字体大小为18
plt.ylabel('Time of run(ms)', fontsize=18)  # 调整y轴标签字体大小为18
plt.title(key[:5] + f' container run time series (less than {duration_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# 格式化文本并添加到 plt.text() 中
plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {duration_mean:<.2f}\n{variance_label:<{max_label_length}} {duration_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url +f"run_time_sequence_less_{duration_threshold}_ms.png")
plt.close()



# 到达时刻的时序图
fig = plt.figure(figsize=(15, 10))
plt.plot(range(len(sequence)), sequence)
plt.xlabel('Arrival order', fontsize=18)  # 调整x轴标签字体大小为18
plt.ylabel('Time of arrival(ms)', fontsize=18)  # 调整y轴标签字体大小为18
total_data_number = len(sequence)
plt.title(key[:5] + f' container arrival time series (total {total_data_number:.2e})', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig(figure_url +"arrival_sequence.png")
plt.close()

# 运行时间的时序图
duration_mean = np.mean(duration)
duration_variance = np.var(duration)
fig = plt.figure(figsize=(15, 10))
plt.plot(range(len(duration)), duration)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
plt.axhline(y=duration_mean, color='r', linestyle='--') # 添加均值线
plt.text(-0.08*xmax, duration_mean, f'{duration_mean:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.xlabel('Arrival order', fontsize=18)  # 调整x轴标签字体大小为18
plt.ylabel('Time of run(ms)', fontsize=18)  # 调整y轴标签字体大小为18
total_data_number = len(duration)
plt.title(key[:5] + f' container run time series (total {total_data_number:.2e})', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# 格式化文本并添加到 plt.text() 中
plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {duration_mean:<.2f}\n{variance_label:<{max_label_length}} {duration_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url +"run_time_sequence.png")
plt.close()

# 运行时间的累积分布图
point_x = 150    ###
# limit_xmax = float('inf')   ###
limit_xmax =400      ###
hist, bin_edges = np.histogram(duration, bins=range(min(duration), max(duration) + 1), density=True)
cumulative_probs = np.cumsum(hist)
index = np.where(bin_edges >= point_x)[0][0]
cumulative_prob_at_point = cumulative_probs[index]
# 绘制累积分布图
fig = plt.figure(figsize=(15, 10))
plt.hist(duration, bins=range(min(duration), max(duration) + 1), cumulative=True, density=True, histtype='step', linewidth=1.5)
plt.xlabel('Arrival order', fontsize=18)
plt.ylabel('Cumulative Probability',fontsize=18)
plt.title(key[:5] + f' Container run time cumulative distribution(less than {limit_xmax} ms)',fontsize=18)
plt.xlim(xmin=min(duration), xmax=min(limit_xmax, max(duration)))  # 限制x轴范围
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 添加init_time对应的垂直虚线和水平虚线
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
plt.axvline(x=point_x, ymin=0, ymax=cumulative_prob_at_point/ymax, color='gray', linestyle='--')  # 添加垂直虚线
plt.axhline(y=cumulative_prob_at_point, xmin= 0,xmax=4000/xmax, color='gray', linestyle='--')  # 添加水平虚线
plt.text(point_x, cumulative_prob_at_point, f'({point_x},{cumulative_prob_at_point:.2f})', fontsize=18, color='black', ha='left', va='top')  # 添加对应的坐标
plt.scatter(point_x, cumulative_prob_at_point, color='gray', s=20)  # 描绘对应的点
# 添加水平虚线和对应的垂直虚线
y_value_max = max(cumulative_probs)
for y_value in np.arange(0.1,y_value_max, 0.1):
    x_value = bin_edges[np.where(cumulative_probs >= (y_value))[0][0]]
    plt.axhline(y=y_value, xmin= 0, xmax=x_value/xmax, color='gray', linestyle='--')  # 添加水平虚线
    plt.axvline(x=x_value, ymin=0, ymax=y_value/ymax, color='gray', linestyle='--')  # 添加对应的垂直虚线
    plt.text(x_value, y_value, f'({x_value},{y_value:.2f})', fontsize=18, color='black', ha='left', va='top')  # 添加对应的坐标
    plt.scatter(x_value, y_value, color='gray', s=20)  # 描绘对应的点
fig.savefig(figure_url + f"run_time_cumulative_distribution_less_{limit_xmax}_ms.png")
plt.close()

# 运行时间的分布图
proportion = 0.99  ###
sorted_duration = sorted(duration)
proportion_index = int(len(sorted_duration) * proportion)
selected_duration = sorted_duration[:proportion_index]
# 设置 x 轴的限制
xlim_min = min(selected_duration)
xlim_max = max (selected_duration)
# 统计直方图
hist, bin_edges = np.histogram(selected_duration, bins=int(max(selected_duration)-min(selected_duration)))
# 找到频率最高的五个 bin
top5_indices = np.argsort(hist)[-5:]
top5_indices = top5_indices[::-1]
top5_bins = bin_edges[top5_indices + 1]  # 使用 +1 是因为 bin_edges 比 hist 长度多一个
fig = plt.figure(figsize=(15, 10))
plt.hist(selected_duration, bins=int(max(selected_duration)-min(selected_duration)))
plt.xlabel('run time(ms)', fontsize=18)
plt.ylabel(f'frequency', fontsize=18)
plt.title(key[:5] + f' Container run time distribution ({xlim_min}-{xlim_max} ms,{proportion:.0%})', fontsize=18)
# 设置 x 轴的限制
plt.xlim(xlim_min, xlim_max)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 绘制最高频率的五个点的坐标
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
tem =0
plt.text(0.85*xmax, 0.95*ymax, f'Top 5: ', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
for bin_center, frequency in zip(top5_bins, hist[top5_indices]):
    plt.scatter(bin_center, frequency, color='gray', marker='o', s=20)  # 绘制点
    plt.text(0.85*xmax, 0.9*ymax-tem*0.05*ymax, f'({bin_center}, {frequency})', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
    tem = tem+1
fig.savefig(figure_url + f"run_time_distribution_{xlim_min}-{xlim_max}_ms.png")
plt.close()

# 到达间隔的时序图，不限制，查看整体
dotted_line_y = 2000   ###
diff_interval_threshold = float('inf')
diff_sequence_mean = np.mean(diff_sequence)
diff_sequence_variance = np.var(diff_sequence)
fig = plt.figure(figsize=(15, 10))
# 布尔索引，选择y轴小于diff_interval_threshold的部分
selected_diff_sequence = [diff for diff in diff_sequence if diff < diff_interval_threshold]
selected_indices = [i for i, diff in enumerate(diff_sequence) if diff < diff_interval_threshold]
plt.plot(selected_indices, selected_diff_sequence)

# plt.axhline(y = dotted_line_y, color='r', linestyle='--')
# plt.text(0, dotted_line_y, f'{dotted_line_y}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标

plt.xlabel('order', fontsize=18)
plt.ylabel('arrival time interval(ms)',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval sequence (less than {diff_interval_threshold} ms)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
# 确定 "mean:" 和 "variance:" 的文本长度
mean_label = "mean: "
variance_label = "variance: "
max_label_length = max(len(mean_label), len(variance_label))
# 格式化文本并添加到 plt.text() 中
plt.text(xmax*0.75, ymax*0.95, f'{mean_label:<{max_label_length}} {diff_sequence_mean:<.2f}\n{variance_label:<{max_label_length}} {diff_sequence_variance:<.2f}', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
fig.savefig(figure_url + f"diff_sequence_less_{diff_interval_threshold}_ms.png")
plt.close()

# 到达间隔的累积分布图
# 确定 init_time 和 mean_run_time 对应的累积概率
limit_xmax = 100    ###
hist, bin_edges = np.histogram(diff_sequence, bins=range(min(diff_sequence), max(diff_sequence) + 1), density=True)
cumulative_probs = np.cumsum(hist)
init_time_index = np.where(bin_edges >= init_time)[0][0]
cumulative_prob_at_init_time = cumulative_probs[init_time_index] # init_time 对应的累积概率
mean_run_time = np.mean(duration)
mean_run_time_index = np.where(bin_edges >= mean_run_time)[0][0]
cumulative_prob_at_mean_run_time = cumulative_probs[mean_run_time_index] # mean_run_time 对应的累积概率
# 绘制累积分布图
fig = plt.figure(figsize=(15, 10))
plt.hist(diff_sequence, bins=range(min(diff_sequence), max(diff_sequence) + 1), cumulative=True, density=True, histtype='step', linewidth=1.5)
plt.xlabel('arrival time interval(ms)',fontsize=18)
plt.ylabel('Cumulative Probability',fontsize=18)
plt.title(key[:5] + f' Container arrival time interval cumulative distribution(less than {limit_xmax}ms)',fontsize=18)
plt.xlim(xmin=min(diff_sequence), xmax=min(limit_xmax, max(diff_sequence)))  # 限制x轴范围
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 添加init_time对应的垂直虚线和水平虚线
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
plt.axvline(x=init_time, ymin=0, ymax=cumulative_prob_at_init_time/ymax, color='r', linestyle='--')  # 添加垂直虚线
plt.axhline(y=cumulative_prob_at_init_time, xmin= 0,xmax=init_time/xmax, color='r', linestyle='--')  # 添加水平虚线
plt.text(init_time, cumulative_prob_at_init_time, f'({init_time},{cumulative_prob_at_init_time:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
plt.text(init_time, -0.01, f'init_time:{init_time}', fontsize=15, color='black', ha='center', va='top')  # 添加对应的x轴坐标
plt.text(0, cumulative_prob_at_init_time, f'{cumulative_prob_at_init_time:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.scatter(init_time, cumulative_prob_at_init_time,color='r', s=20)  # 描绘对应的点
# 添加mean_run_time对应的垂直虚线和水平虚线
plt.axvline(x=mean_run_time, ymin=0, ymax=cumulative_prob_at_mean_run_time/ymax, color='g', linestyle='--')  # 添加垂直虚线
plt.axhline(y=cumulative_prob_at_mean_run_time, xmin= 0,xmax=mean_run_time/xmax, color='g', linestyle='--')  # 添加水平虚线
plt.text(mean_run_time, cumulative_prob_at_mean_run_time, f'({mean_run_time:.2f},{cumulative_prob_at_mean_run_time:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
plt.text(mean_run_time, 0.04, f'mean_run_time:{mean_run_time:.2f}', fontsize=15, color='black', ha='center', va='top')  # 添加对应的x轴坐标
plt.text(0, cumulative_prob_at_mean_run_time, f'{cumulative_prob_at_mean_run_time:.2f}', fontsize=15, color='black', ha='right', va='center')  # 添加对应的y轴坐标
plt.scatter(mean_run_time, cumulative_prob_at_mean_run_time, color='g', s=20)  # 描绘对应的点
# 每10%描出坐标，添加水平虚线和对应的垂直虚线
plt.axhline(y=1, color='gray', linestyle='--')  # 添加水平虚线
y_value_max = max(cumulative_probs)
for y_value in np.arange(0.5,y_value_max, 0.1):
    x_value = bin_edges[np.where(cumulative_probs >= (y_value))[0][0]]
    plt.axhline(y=y_value, xmin= 0, xmax=x_value/xmax, color='gray', linestyle='--')  # 添加水平虚线
    plt.axvline(x=x_value, ymin=0, ymax=y_value/ymax, color='gray', linestyle='--')  # 添加对应的垂直虚线
    plt.text(x_value, y_value, f'({x_value},{y_value:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
    plt.scatter(x_value, y_value, color='gray', s=20)  # 描绘对应的点
fig.savefig(figure_url + f"diff_cumulative_distribution_less_{limit_xmax}_ms.png")
plt.close()

# 到达间隔的分布图
proportion = 0   ###
limit_xmax = 30050    ###
limit_xmin = 29980    ###
sorted_diff_sequence = sorted(diff_sequence)
proportion_index = int(len(sorted_diff_sequence) * proportion)
selected_diff_sequence = sorted_diff_sequence[proportion_index:]
# 统计直方图
hist, bin_edges = np.histogram(selected_diff_sequence, bins=int(max(selected_diff_sequence)-min(selected_diff_sequence)))
# 找到频率最高的五个 bin
top5_indices = np.argsort(hist)[-5:]
top5_indices = top5_indices[::-1]
top5_bins = bin_edges[top5_indices + 1]  # 使用 +1 是因为 bin_edges 比 hist 长度多一个
fig = plt.figure(figsize=(15, 10))
plt.hist(selected_diff_sequence, bins=int(max(selected_diff_sequence)-min(selected_diff_sequence)))
plt.xlabel('arrival time interval(ms)', fontsize=18)
plt.ylabel(f'frequency', fontsize=18)
# plt.title(key[:5] + f' Container arrival time interval distribution ({proportion:.0%})', fontsize=18)
plt.title(key[:5] + f' Container arrival time interval distribution ({limit_xmin}-{limit_xmax}ms)', fontsize=18)
plt.xlim(xmin=max(limit_xmin, min(selected_diff_sequence)), xmax=min(limit_xmax, max(selected_diff_sequence)))  # 限制x轴范围
# plt.text(init_time, 5, f'init_time:{init_time}', fontsize=15, color='black', ha='center', va='top')  # 添加init_time对应的x轴坐标
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 绘制最高频率的五个点的坐标
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
tem =0
plt.text(xmin+0.85*(xmax-xmin), 0.95*ymax, f'Top 5: ', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
for bin_center, frequency in zip(top5_bins, hist[top5_indices]):
    plt.scatter(bin_center, frequency, color='gray', marker='o', s=20)  # 绘制点
    plt.text(xmin+0.85*(xmax-xmin), 0.9*ymax-tem*0.05*ymax, f'({bin_center}, {frequency})', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
    tem = tem+1
# fig.savefig(figure_url + f"interval_distribution_{proportion:.0%}.png")
fig.savefig(figure_url + f"interval_distribution_{limit_xmin}-{limit_xmax}_ms.png")
plt.close()
# 到达间隔的分布图
proportion = 0.8
sorted_diff_sequence = sorted(diff_sequence)
proportion_index = int(len(sorted_diff_sequence) * proportion)
selected_diff_sequence = sorted_diff_sequence[proportion_index:]
# 统计直方图
hist, bin_edges = np.histogram(selected_diff_sequence, bins=int(max(selected_diff_sequence)-min(selected_diff_sequence)))
# 找到频率最高的五个 bin
top5_indices = np.argsort(hist)[-5:]
top5_indices = top5_indices[::-1]
top5_bins = bin_edges[top5_indices + 1]  # 使用 +1 是因为 bin_edges 比 hist 长度多一个
fig = plt.figure(figsize=(15, 10))
plt.hist(selected_diff_sequence, bins=int(max(selected_diff_sequence)-min(selected_diff_sequence)))
plt.xlabel('arrival time interval(ms)', fontsize=18)
plt.ylabel(f'frequency', fontsize=18)
plt.title(key[:5] + f' Container arrival time interval distribution ({proportion:.0%})', fontsize=18)
# plt.text(init_time, 5, f'init_time:{init_time}', fontsize=15, color='black', ha='center', va='top')  # 添加init_time对应的x轴坐标
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 绘制最高频率的五个点的坐标
ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()
tem =0
plt.text(0.85*xmax, 0.95*ymax, f'Top 5: ', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
for bin_center, frequency in zip(top5_bins, hist[top5_indices]):
    plt.scatter(bin_center, frequency, color='gray', marker='o', s=20)  # 绘制点
    plt.text(0.85*xmax, 0.9*ymax-tem*0.05*ymax, f'({bin_center}, {frequency})', fontsize=15, color='black', ha='left', va='bottom')  # 添加对应的坐标
    tem = tem+1
fig.savefig(figure_url + f"interval_distribution_{proportion:.0%}.png")
plt.close()
