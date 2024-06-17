#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
保活五分钟
exe={'key':[[start_time,request_use_time],...],...}
未改，有问题，影响了计算
修改Platform中statistics文件中的memory_statistics函数,sys_use_time(即all_time) = request_use_time + waste_time
改为

'''
from Platform import input,statistics
import matplotlib.pyplot as plt
import os

def keep5min(requests,metas,keep_time=300000):
    exe=input.createdict(metas)
    warm=input.createdict(metas)
    cold_start=input.createdict(metas)
    waste_time=input.createdict(metas)
    exe_time=input.createdict(metas)
    response_fail=input.createdict(metas)

    num = 1
    request_end_time=0


    for req in requests:
        now_time=req["startTime"]
        req_key=req["metaKey"]
        meta= input.getmeta(req, metas)

        #更新exe warm,该时刻前已结束的容器需要移除
        for meta_key in exe:
            containers=exe[meta_key]
            exeindex=0
            while  exeindex < len(containers) and containers[exeindex]<=now_time:
                exeindex=exeindex+1
            finish=containers[0:exeindex]
            exe[meta_key]=containers[exeindex:]
            finish = [x + keep_time for x in finish]
            warm[meta_key].sort()
            warm[meta_key].extend(finish)

        for meta_key in warm:
            warm[meta_key].sort()
            containers = warm[meta_key]
            warmindex = 0
            while  warmindex < len(containers) and containers[warmindex]<=now_time:
                warmindex=warmindex+1
            warm[meta_key] = containers[warmindex:]
            
            tem = [keep_time]*warmindex
            waste_time[meta_key].extend(tem)


        #判断冷热启动
        if req_key in warm and warm[req_key]!=[]:
            #热启动
            init_time=0
            #更新exe warm
            exe[req_key].append(req["startTime"]+init_time+req["durationsInMs"])
            exe[req_key].sort()
            warm[meta_key].sort()
            warm_end=warm[req_key].pop(0)
            waste_time[req_key].append(keep_time-(warm_end-now_time))
        else:
            #冷启动
            init_time=meta["initDurationInMs"]
            exe[req_key].append(req["startTime"]+init_time + req["durationsInMs"])
            exe[req_key].sort()
            cold_start[req_key].append(now_time)

        #检查是否可以正常响应
        if init_time+req["durationsInMs"]>meta["timeoutInSecs"]*1000:
            #不能正常响应 exe更新
            exe[req_key].remove(req["startTime"]+init_time + req["durationsInMs"])
            warm[meta_key].sort()
            warm[req_key].append(req["startTime"]+init_time + req["durationsInMs"]+keep_time)
            # 统计  响应失败
            response_fail[req_key].append(now_time)
            request_end_time =req["startTime"] + meta["timeoutInSecs"]*1000
            exe_time[req_key].append([init_time+req["durationsInMs"],init_time+req["durationsInMs"]])
        else:
            # 正常响应
            request_end_time = req["startTime"] + init_time + req["durationsInMs"]
            exe_time[req_key].append([init_time+req["durationsInMs"],init_time+req["durationsInMs"]])
        # print("完成第",num,"个请求")
        num=num+1
    # 未关闭的热启动容器记录waste_time
    for meta_key in warm:
        warm[meta_key].sort()
        warm[meta_key]=[(keep_time-(ele-request_end_time)) for ele in warm[meta_key]]
        waste_time[meta_key].extend(warm[meta_key])
    # print(cold_start)
    # print(waste_time)
    return [cold_start,waste_time,exe_time,response_fail]


if __name__=="__main__":



    # [requests,metas]=input.input(r"/home/wangyi/serverless/test_data")
    [requests,metas]=input.input("/home/wangyi/serverless/dataSet_2")
    # key = 'roles2'
    key = 'nodes2'
    # key = 'rolebindings2'
    # key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
    # key = 'nodes1'
    figure_url = '/home/wangyi/serverless/keepalive_search_figures/' + key + '/'
    if not os.path.exists(figure_url):
        os.makedirs(figure_url)
    start = 25000
    end = 35000
    step = 1000
    time_range = range(start,end,step)
    time_utilization=[]
    memory_utilization=[]
    mean_utilization=[]
    for keep_time in time_range:
        print(f'{keep_time}/{end}')
        [cold_start,waste_time,exe_time,response_fail]=keep5min(requests,metas,keep_time)
        cold_statistics=statistics.cold_start_statistics(cold_start,exe_time,metas)
        mem_statistics=statistics.memory_statistics(waste_time,cold_start,exe_time,metas)
        time_utilization.append(cold_statistics[key][-2])
        memory_utilization.append(mem_statistics[key][-1])
        mean_utilization.append((cold_statistics[key][-2]+mem_statistics[key][-1])/2)
    # 最佳时间的索引
    point_index = mean_utilization.index(max(mean_utilization))
    #　画图
    fig = plt.figure()
    plt.plot(time_range,mean_utilization,label='mean of utilizations')
    plt.plot(time_range,time_utilization,label='time utilization')
    plt.plot(time_range,memory_utilization,label='memory utilization')
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    # 添加对应的垂直虚线和水平虚线

    plt.axvline(x=time_range[point_index], ymin=0, ymax=(time_utilization[point_index]-ymin)/(ymax-ymin), color='gray', linestyle='--')  # 添加垂直虚线
    plt.axhline(y=time_utilization[point_index], xmin= 0, xmax=(time_range[point_index]-xmin)/(xmax-xmin), color='gray', linestyle='--')  # 添加水平虚线
    plt.text(time_range[point_index], time_utilization[point_index], f'({time_range[point_index]},{time_utilization[point_index]:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标
    
    plt.axhline(y=mean_utilization[point_index], xmin= 0,xmax=(time_range[point_index]-xmin)/(xmax-xmin), color='gray', linestyle='--')  # 添加水平虚线
    plt.text(time_range[point_index], mean_utilization[point_index], f'({time_range[point_index]},{mean_utilization[point_index]:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标

    plt.axhline(y=memory_utilization[point_index], xmin=0,xmax=(time_range[point_index]-xmin)/(xmax-xmin), color='gray', linestyle='--')  # 添加水平虚线
    plt.text(time_range[point_index], memory_utilization[point_index], f'({time_range[point_index]},{memory_utilization[point_index]:.2f})', fontsize=15, color='black', ha='left', va='top')  # 添加对应的坐标

    plt.xlabel('time(ms)')
    plt.ylabel('utilization(%)')
    plt.title( key + ' keep alive time search')
    plt.legend()
    plt.savefig( figure_url + key + f'_keepalive_search_{start}_{end}_{step}.png')


    # res_statistics=statistics.response_statistics(response_fail,exe_time,metas)
    # TODO 输出结果 导入表格
    # print('cold_statistics:')
    # print('cold_statistics[key]=[cold_num, all_num, frequency, cold_time, all_time, utilization, cold_time_every_req]')
    # for key in cold_statistics:
    #     print(key,":",cold_statistics[key])
    # print('mem_statistics:')
    # print('mem_statistics[key]=[waste_mem_every_req, waste_mem, req_exe_mem_every_req, req_exe_mem, all_mem, utilization]')
    # for key in mem_statistics:
    #     print(key,":",mem_statistics[key])
    # print('res_statistics:',res_statistics)
