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

def keep5min(requests,metas,keep_time):
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
    # requests = requests[-int(1/6*len(requests)):]
    # key = '8371b8baba81aac1ca237e492d7af0d851b4d141'
    key = 'nodes2'
    keep_time = 0

    [cold_start,waste_time,exe_time,response_fail]=keep5min(requests,metas,keep_time)

    cold_statistics=statistics.cold_start_statistics(cold_start,exe_time,metas)
    mem_statistics=statistics.memory_statistics(waste_time,cold_start,exe_time,metas)
    # res_statistics=statistics.response_statistics(response_fail,exe_time,metas)
    # TODO 输出结果 导入表格
    print('cold_statistics:')
    print('cold_statistics[key]=[cold_num, all_num, frequency, cold_time, all_time, utilization, cold_time_every_req]')
    # for key in cold_statistics:
    print(key,":",cold_statistics[key])
    print('mem_statistics:')
    print('mem_statistics[key]=[waste_mem_every_req, waste_mem, req_exe_mem_every_req, req_exe_mem, all_mem, utilization]')
    # for key in mem_statistics:
    print(key,":",mem_statistics[key])
    # print('res_statistics:',res_statistics)
