import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


def read_test(beta=100000000):
    file = open('run_total.log', 'r')

    lines = file.readlines()
    L1RdCount = []
    L1WrCount = []
    L2RdCount = []
    L2WrCount = []

    alpha = 6


    # num = 0
    for line in lines:
        if line.find('L1RdCount:') >= 0:
            num = [int(s) for s in line.split() if s.isdigit()][0]
            L1RdCount.append(num)
        if line.find('L1WrCount:') >= 0:
            num = [int(s) for s in line.split() if s.isdigit()][0]
            L1WrCount.append(num)
        if line.find('L2RdCount:') >= 0:
            num = [int(s) for s in line.split() if s.isdigit()][0]
            L2RdCount.append(num)
        if line.find('L2WrCount:') >= 0:
            num = [int(s) for s in line.split() if s.isdigit()][0]
            L2WrCount.append(num)
        # if line.find('GFLOPS with') >= 0:
        #     num = float(line.split()[2])
        # if line.find('Compute rumtime') >= 0:
        #     time = float(line.split()[2])
        #     comp_time.append(time * num)
    
   # for i in range(28):
        #print('layer type', i+1)
        #print('l1rd: ', L1RdCount[i], L1WrCount[i], L2RdCount[i], L2WrCount[i])    
        #print('l1wr: ', L1WrCount[i])    
        #print('l2rd: ', L2RdCount[i])    
        #print('l2wr: ', L2WrCount[i])    

    df = pd.read_excel('./model_ResNet50.xlsx', sheetname='ResNet50')

    # print("Column headings:")
    layers = []
    bottlenecks = []
    comp_time_block = 0
    mem_time_block = 0
    bottlenecks_comp = []
    for i, item in enumerate(df['Layer name']):
        if i <= 1 or i >= 70:
            continue 

        if 'ds' in item:
            continue

        if item == 'Sum':

            bottlenecks_comp.append(comp_time_block / beta)
            bottlenecks.append((comp_time_block + mem_time_block) / beta)
            comp_time_block = 0
            mem_time_block = 0
            continue
        index = int(df['layer type'][i])-1
        ofmap_h = df[' OFMAP Height'][i]
        ofmap_w = df[' OFMAP Width'][i]
        channel = df[' Channels'][i]
        filter_h = df[' Filter Height'][i]
        filter_w = df[' Filter Width'][i]
        filter_num = df[' Num Filter'][i]
        comp_time_tmp = ofmap_h * ofmap_w * channel * filter_h * filter_w * filter_num
        comp_time_block += comp_time_tmp
        mem_access_tmp = L1RdCount[index] + L1WrCount[index] + alpha * L2RdCount[index] + alpha * L2WrCount[index]
        mem_time_block += mem_access_tmp
        # layers.append(comp_time_tmp)

    # print(bottlenecks)
    # print(bottlenecks_comp)
    return bottlenecks
# read_test()

