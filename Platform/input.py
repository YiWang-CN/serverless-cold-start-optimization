#!/usr/bin/env python 
# -*- coding:utf-8 -*-

'''
打开该路径下的两个文件
建立两个列表，按时间排序
根据request获取对应meta

### requests 数据集
requests 数据集包含了一系列的请求记录，每条记录对应一个已经完成的任务。每条记录包含以下字段：
*** startTime: 这是一个以毫秒为单位的时间戳，代表任务开始的时间。
* metaKey: 用来标识与任务相关联的元数据，它与元数据数据集中的key字段相对应。
*** durationsInMs: 表示任务的执行时间，单位为毫秒。
* statusCode: 表示任务的执行结果。一般来说，200代表任务成功完成，其他值则可能代表有错误或异常情况，模拟的是云计算场景下后端实例偶发性的异常。

### metas 数据集
metas 数据集包含了一系列的元数据记录，每条记录包含一项任务的配置信息。每条记录包含以下字段：
* key: 用来唯一标识一项元数据,可以理解为任务的类型或者特征。它与requests数据集中的metaKey字段相对应。
*** runtime: 代表任务需要运行的环境，如python、nodejs、go等。
*** memoryInMb: 表示任务需要的内存数量，单位为MB。
*** timeoutInSecs: 表示任务的超时时间，单位为秒。
*** initDurationInMs: 表示任务初始化的时间，单位为毫秒。'''

import json

def input(filepath):
    # 读取requests
    with open(filepath+r'/requests', 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
    # 初始化一个空列表用于存储解析后的数据
    requests = []
    # 遍历每一行数据
    for line in lines:
        # 解析JSON格式的数据并添加到列表中
        data = json.loads(line)
        requests.append(data)

    # 读取metas
    with open(filepath+r'/metas', 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
    # 初始化一个空列表用于存储解析后的数据
    metas = []
    # 遍历每一行数据
    for line in lines:
        # 解析JSON格式的数据并添加到列表中
        data = json.loads(line)
        metas.append(data)
    # print(metas)

    # 获取requests中第二项startTime的值
    def takestartTime(elem):
        return elem['startTime']
    # 对requests列表按照时隙先后排序
    requests.sort(key=takestartTime)
    # print(requests)

    return [requests,metas]

# 获取函数调用的资源需求
def getmeta(request,metas):
    key=request['metaKey']
    for meta in metas:
        if key==meta['key']:
            return meta

#创建字典包含所有的容器类型，对应键的值为[]
def createdict(metas):
    ditc={}
    for ele in metas:
        ditc.setdefault(ele["key"],[])
    return ditc

if __name__ =="__main__":
    input(r"F:\python_file\serverless\dataSet_1")
