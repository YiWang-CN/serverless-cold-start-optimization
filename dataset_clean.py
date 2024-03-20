
from Platform import input

'''清理 数据集 中难以利用的数据，删除请求数量<10000，的容器'''



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

dataset_name = 'dataSet_1'
[requests, metas] = input.input(r"/home/wangyi/serverless/" + dataset_name)
sequences = get_sequences(requests, metas)

for key in sequences:
    sequence = sequences[key]
    if len(sequence) <= 10000:
        remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/requests", key)
        remove_lines_with_string(r"/home/wangyi/serverless/" + dataset_name + "/metas", key)
