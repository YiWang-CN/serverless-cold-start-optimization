#!/usr/bin/env python 
# -*- coding:utf-8 -*-

'''
指标统计：冷启动、资源利用、响应
冷启动：每种容器的冷启动次数，总次数，冷启动频率，冷启动的时延，时间利用率
cold_start={'key':[start_time,...],...}
exe_time={'key':[[start_time,use_time],...],...}
metas=metas
资源利用：每种容器的浪费资源，总使用资源，资源利用率
waste_time{'key':[container_free_time,...],...}
响应：每种容器的失败响应次数，响应次数，响应率
response_fail={'key':[start_time,...],...}
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
        cold_statistics[key]=[cold_num,all_num,frequency,init_time,utilization]

        all_all_time=all_all_time+all_time
        all_cold_time=all_cold_time+cold_num*init_time

    all_frequency = all_cold_num/all_all_num
    all_utilization = 1-all_cold_time/all_all_time
    cold_statistics['all']=[all_cold_num,all_all_num,all_frequency,all_utilization]
    return cold_statistics

# 记录资源利用，输出一个字典，记录每种容器的浪费资源，总使用资源，资源利用率
def memory_statistics(waste_time,exe_time,metas):
    mem_statistics=input.createdict(metas)
    all_waste_mem=0
    all_all_mem=0

    for key in waste_time:
        waste=sum(waste_time[key])

        all_time = 0
        for row in exe_time[key]:
            all_time = all_time+row[1]
        all_time=all_time+waste

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

# 记录响应，输出一个字典，记录每种容器的失败响应次数，响应次数，响应率
def response_statistics(response_fail,exe_time,metas):
    res_statistics = input.createdict(metas)
    all_fail_num=0
    all_res_num=0

    for key in response_fail:
        fail_num=len(response_fail[key])
        res_num=len(exe_time[key])
        all_fail_num=all_fail_num+fail_num
        all_res_num=all_res_num+res_num
        if res_num == 0:
            response_rate = 0
        else:
            response_rate=1-(fail_num/res_num)
        res_statistics[key]=[fail_num,res_num,response_rate]

    all_response_rate =1-(all_fail_num/all_res_num)
    res_statistics['all']=[all_fail_num,all_res_num,all_response_rate]
    return res_statistics
# todo 输出结果 导入表格