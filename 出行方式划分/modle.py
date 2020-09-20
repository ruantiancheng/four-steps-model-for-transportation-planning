import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

arr = np.array(range(1, 17))
dataframe = pd.DataFrame(arr.reshape(8, 2), columns=['gongjiao', 'xiaoqiche'])
arr = np.ones((6, 6)) * 85
traffic_cost = pd.DataFrame(arr, columns=range(1, 7), index=range(1, 7))
print(traffic_cost)


def curel_fit(traffic_cost, exp_cartime=[75, 80, 85, 90, 100], exp_bustime=[85, 95, 110, 130, 160]):
    exp_cartime = np.array(exp_cartime)
    exp_bustime = np.array(exp_bustime)
    fit_fuc = np.polyfit(exp_cartime, exp_bustime, 3)
    bustime_cal = np.poly1d(fit_fuc)
    exp_bustimevals = bustime_cal(exp_cartime)
    bus_traffic_cost = traffic_cost.apply(lambda x: bustime_cal(x))
    index_list = []
    for i in range(1, 1 + len(traffic_cost)):
        index_list.append('bus%i' % i)
    bus_traffic_cost.columns = index_list
    new_traffic_cost = pd.concat([traffic_cost, bus_traffic_cost], axis=1)
    return new_traffic_cost


newtraffic_cost = curel_fit(traffic_cost)


# 曲线可视化部分
# plot1 = plt.plot(exp_cartime, exp_bustimevals, 'r', label='original values')
# plt.title('polyfitting')
# plt.show()

def logit(dataframe):
    # logit 模型算法

    split_choice = []
    lens = len(dataframe)
    columns = len(dataframe.columns)


    for i in range(lens):
        for j in range(int(columns/2)):
            sum_list = []
            for k in range(1,1+int(columns/lens)):
                real_resist = math.e ** -(dataframe.iloc[i,k*lens-1]/50)
                sum_list.append(real_resist)

            split_choice.append(math.e ** -(dataframe.iloc[i,j]/50 )/ sum(sum_list))
    split_choice = np.array(split_choice)
    split_choice = pd.DataFrame(split_choice.reshape(lens, int(columns / 2)))
    bus_split_choice = split_choice.apply(lambda x:1-x)
    split_choice = pd.concat([split_choice,bus_split_choice],axis=1)
    index_list = []
    for k in range(1,lens+1):
        index_list.insert(k-1,'car%i'%k)
        index_list.append('bus%i'%k)
    split_choice.columns=index_list
    split_choice.index = range(1,lens+1)


    return split_choice


print(logit(newtraffic_cost))
