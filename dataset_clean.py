'''清理 数据集 中难以利用的数据，
删除请求数量<10000，的容器,
请求间隔的1/3极差大于均值视为存在噪音，
将请求间隔大于【EX+2σ且大于1/3极差】的请求视为请求已经脱离了到来规律，依据此类的请求间隔为分割线，划分区间，
区间内请求数量大于【2*train_window且大于区间大小均值】的区间视为正常的请求到来。
将该容器的正常区间分别写入serverless/dataset/container_key/request1,request2...文件。'''
from Platform import input
import statistics





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

# 定义函数，删除与字符串匹配的行
def remove_lines_with_string(file_path, string):
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 过滤掉包含特定字符串的行
    filtered_lines = [line for line in lines if string not in line]
    
    # 写回文件
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# 定义函数，计算时间序列的间隔
def calculate_interval(sequence):
    intervals = []
    for i in range(1, len(sequence)):
        interval = sequence[i] - sequence[i-1]
        intervals.append(interval)
    return intervals

# 请求间隔的1/3极差大于均值视为存在噪音，
# 将请求间隔【大于100秒】的请求视为请求已经脱离了到来规律，依据此类的请求间隔为分割线，划分区间，
# 区间内请求数量【大于2*train_window且大于区间大小均值】的区间视为正常的请求到来。
# 将该容器的正常区间分别写入serverless/dataset/container_key/request1,request2...文件。

def clean_dataset(dataset_name, train_window):
    [requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
    sequences = get_sequences(requests, metas)
    for key in sequences:
        sequence = sequences[key]
        intervals = calculate_interval(sequence)
        mean = statistics.mean(intervals)
        std_dev = statistics.stdev(intervals)
        third_range = (max(intervals) - min(intervals)) / 3
        threshold = mean + 2 * std_dev

        if third_range > mean: # 请求间隔的1/3极差大于均值视为存在噪音
            interval_outliers = [i for i, interval in enumerate(intervals) if interval > 10**5]
            
            interval_ranges = []
            sequence_rangs = []
            current_range = []
            current_sequence_range = []
            for i in range(len(intervals)):
                if i in interval_outliers:
                    if current_range:
                        interval_ranges.append(current_range)
                        current_range = []
                    # if current_sequence_range:
                    #     sequence_rangs.append(current_sequence_range)
                    #     current_sequence_range = []
                else:
                    current_range.append(intervals[i])
                    # current_sequence_range.append(sequence[i+1])
            if current_range:
                interval_ranges.append(current_range)
            # if current_sequence_range:
            #     sequence_rangs.append(current_sequence_range)



            for i, interval_range in enumerate(interval_ranges):
                if len(interval_range) > 2 * train_window and len(interval_range) > statistics.mean(interval_range):
                    with open(f"request{i+1}.txt", "w") as file:
                        file.write(f"Request interval range {i+1}:\n")
                        file.write(f"Start Time\tEnd Time\tInterval\n")
                        for j in range(len(interval_range)):
                            file.write(f"{sequence[j]}\t{sequence[j+1]}\t{interval_range[j]}\n")


                            
            if threshold > third_range:
                remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/requests", key)
                remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/metas", key)
        else:
            interval_ranges = []
            current_range = []
            for interval in intervals:
                if interval > threshold and interval > third_range:
                    if current_range:
                        interval_ranges.append(current_range)
                        current_range = []
                current_range.append(interval)
            if current_range:
                interval_ranges.append(current_range)
            for i, interval_range in enumerate(interval_ranges):
                if len(interval_range) > 2 * train_window and len(interval_range) > statistics.mean(interval_range):
                    with open(f"request{i+1}.txt", "w") as file:
                        file.write(f"Request interval range {i+1}:\n")
                        file.write(f"Start Time\tEnd Time\tInterval\n")
                        for j in range(len(interval_range)):
                            file.write(f"{sequence[j]}\t{sequence[j+1]}\t{interval_range[j]}\n")



dataset_name = 'dataSet_1'
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
sequences = get_sequences(requests, metas)

# 清理数据集，删除请求数量<10000的容器
for key in sequences:
    sequence = sequences[key]
    if len(sequence) <= 10000:
        remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/requests", key)
        remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/metas", key)


        


