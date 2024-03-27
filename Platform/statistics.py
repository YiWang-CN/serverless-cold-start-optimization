#!/usr/bin/env python 
# -*- coding:utf-8 -*-

'''
指标统计：冷启动、资源利用、响应
输入：cold_start/cold_start_predict、waste_time、exe_time/exe_time_predict
冷启动：每种容器的冷启动次数，总次数，冷启动频率，冷启动的时延，时间利用率
cold_start={'key':[start_time,...],...}         cold_start_predict={'key':[[start_time,prepare_time],...],...}
exe_time={'key':[[start_time,request_use_time],...],...}        exe_time_predict={'key':[[sys_use_time,request_use_time],...],...}
metas=metas
资源利用：每种容器的浪费资源，总使用资源，资源利用率
waste_time{'key':[container_free_time,...],...}
响应：每种容器的失败响应次数，响应次数，响应率
response_fail={'key':[start_time,...],...}

指标计算：
冷启动相关：
cold_statistics[key]=[cold_num,all_num,cold_frequency,cold_time,utilization]
cold_time：冷启动的时延，请求到来后仍需准备的时间
utilization：时间利用率，1-冷启动时间/请求使用时间；总时间为请求到来后使用容器的时间总和（准备时间+执行时间）
资源利用相关：
mem_statistics[key]=[waste_mem,all_mem,utilization]
waste_mem：浪费的内存，浪费的时间*容器内存；浪费的时间为热容器空闲时间总和
all_mem：总使用的内存，系统使用的时间*容器内存；系统使用的时间为容器从开始初始化到最终结束的时间总和
'''

from . import input
# 计算冷启动 输出一个字典，记录每种容器的冷启动次数，总次数，冷启动频率，冷启动的时延，时间利用率
def cold_start_statistics(cold_start,exe_time,metas):
    cold_statistics=input.createdict(metas)
    all_cold_num = 0
    all_all_num = 0
    all_all_time = 0
    all_cold_time = 0

    for key in cold_start:
        cold_num=len(cold_start[key])
        all_num=len(exe_time[key])
        all_time = 0
        all_cold_num=all_cold_num+cold_num
        all_all_num=all_all_num+all_num


        if all_num == 0:
            frequency = 0
        else:
            frequency=cold_num/all_num
        init_time=0
        for ele in metas:
           if key == ele['key']:
               init_time = ele["initDurationInMs"]
               break

        for row in exe_time[key]:
            all_time = all_time+row[1]

        if all_time == 0:
            utilization = 0
        else:
            utilization=1-cold_num*init_time/all_time
        cold_statistics[key]=[cold_num,all_num,frequency,cold_num*init_time,utilization]

        all_all_time=all_all_time+all_time
        all_cold_time=all_cold_time+cold_num*init_time

    all_frequency = all_cold_num/all_all_num
    all_utilization = 1-all_cold_time/all_all_time
    cold_statistics['all']=[all_cold_num,all_all_num,all_frequency,all_cold_time,all_utilization]
    return cold_statistics

# 使用预测模式情况下 计算冷启动 输出一个字典，记录每种容器的冷启动次数，总次数，冷启动频率，冷启动的时延，时间利用率
def cold_start_statistics_predict(cold_start_predict,exe_time,metas):
    cold_statistics=input.createdict(metas)
    all_cold_num = 0
    all_all_num = 0
    all_all_time = 0
    all_cold_time = 0

    for key in cold_start_predict:
        cold_num=len(cold_start_predict[key])
        all_num=len(exe_time[key])
        all_time = 0
        cold_time = 0
        all_cold_num=all_cold_num+cold_num
        all_all_num=all_all_num+all_num

        if all_num == 0:
            frequency = 0
        else:
            frequency = cold_num/all_num

        # 修改了冷启动时延的计算方式，有部分冷启动时延不是完全init_time
        for row in cold_start_predict[key]:
            cold_time = cold_time + row[1]

        for row in exe_time[key]:
            all_time = all_time+row[1]

        if all_time == 0:
            utilization = 0
        else:
            utilization=1-cold_time/all_time
        cold_statistics[key]=[cold_num,all_num,frequency,cold_time,utilization]

        all_all_time=all_all_time+all_time
        all_cold_time=all_cold_time+cold_time

    all_frequency = all_cold_num/all_all_num
    all_utilization = 1-all_cold_time/all_all_time
    cold_statistics['all']=[all_cold_num,all_all_num,all_frequency,all_cold_time,all_utilization]
    return cold_statistics

# 记录资源利用，输出一个字典，记录每种容器的浪费资源，总使用资源，资源利用率
def memory_statistics(waste_time,exe_time,metas):
    mem_statistics=input.createdict(metas)
    all_waste_mem=0
    all_all_mem=0

    for key in waste_time:
        waste=sum(waste_time[key])

        # 因为不方便统计容器的总使用时间，所以使用request_use_time + waste_time代替
        all_time = 0
        for row in exe_time[key]:
            all_time = all_time+row[1]
        all_time = all_time + waste

        key_memory = 0
        for ele in metas:
            if key == ele['key']:
                key_memory = ele["memoryInMb"]
                break

        waste_mem=waste*key_memory
        all_mem = all_time*key_memory
        all_waste_mem=all_waste_mem+waste_mem
        all_all_mem=all_all_mem+all_mem

        if all_mem == 0:
            utilization=0
        else:
            utilization=1-waste_mem/all_mem

        mem_statistics[key]=[waste_mem,all_mem,utilization]
    all_utilization = 1 - all_waste_mem / all_all_mem
    mem_statistics['all']=[all_waste_mem,all_all_mem,all_utilization]
    return mem_statistics

# 使用预测模式情况下, 记录资源利用，输出一个字典，记录每种容器的浪费资源，总使用资源，资源利用率
def memory_statistics_predict(waste_time,exe_time,metas):
    mem_statistics=input.createdict(metas)
    all_waste_mem=0
    all_all_mem=0

    for key in waste_time:
        waste=sum(waste_time[key])

        all_time = 0
        for row in exe_time[key]:
            all_time = all_time+row[0]

        key_memory = 0
        for ele in metas:
            if key == ele['key']:
                key_memory = ele["memoryInMb"]
                break

        waste_mem=waste*key_memory
        all_mem = all_time*key_memory
        all_waste_mem=all_waste_mem+waste_mem
        all_all_mem=all_all_mem+all_mem

        if all_mem == 0:
            utilization=0
        else:
            utilization=1-waste_mem/all_mem

        mem_statistics[key]=[waste_mem,all_mem,utilization]
    all_utilization = 1 - all_waste_mem / all_all_mem
    mem_statistics['all']=[all_waste_mem,all_all_mem,all_utilization]
    return mem_statistics

# # 记录响应，输出一个字典，记录每种容器的失败响应次数，响应次数，响应率
# def response_statistics(response_fail,exe_time,metas):
#     res_statistics = input.createdict(metas)
#     all_fail_num=0
#     all_res_num=0

#     for key in response_fail:
#         fail_num=len(response_fail[key])
#         res_num=len(exe_time[key])
#         all_fail_num=all_fail_num+fail_num
#         all_res_num=all_res_num+res_num
#         if res_num == 0:
#             response_rate = 0
#         else:
#             response_rate=1-(fail_num/res_num)
#         res_statistics[key]=[fail_num,res_num,response_rate]

#     all_response_rate =1-(all_fail_num/all_res_num)
#     res_statistics['all']=[all_fail_num,all_res_num,all_response_rate]
#     return res_statistics
