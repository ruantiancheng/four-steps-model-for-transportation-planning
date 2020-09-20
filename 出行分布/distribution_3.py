# -*- coding: utf-8 -*-
'''
    输入：.shp路网文件
            pa矩阵
    功能：1.读取路网文件
        2.计算路径时间
        3.检索最短路径
        4.解决小区到小区问题，现在将小区与小区节点路径修正到原来的节点中
        5.增加了迭代次数
    问题：重力模型里gamma值没有修正，只做了单约束没做双约束导致PA不平衡
* ━━━━━━神兽出没━━━━━━
* 　　　┏┓　　　┏┓
* 　　┏┛┻━━━┛┻┓
* 　　┃　　　　　　　┃
* 　　┃　　　━　　　┃
* 　　┃　┳┛　┗┳　┃
* 　　┃　　　　　　　┃
* 　　┃　　　┻　　　┃
* 　　┃　　　　　　　┃
* 　　┗━┓　　　┏━┛Code is far away from bug with the animal protecting
* 　　　　┃　　　┃                 神兽保佑,代码无bug
* 　　　　┃　　　┃
* 　　　　┃　　　┗━━━┓
* 　　　　┃　　　　　　　┣┓
* 　　　　┃　　　　　　　┏┛
* 　　　　┗┓┓┏━┳┓┏┛
* 　　　　　┃┫┫　┃┫┫
* 　　　　　┗┻┛　┗┻┛
*
* ━━━━━━感觉萌萌哒━━━━━━
'''

import numpy as np
import pandas as pd
import dbfread as dbf
import csv
import shapefile
import matplotlib.pyplot as plt
import geopandas as gdf
import math
import time
import copy
import os


def read_shp(file_name, point_number,
             vot=[4230.8, 3296.3, 3661.525, 3854.55, 3527.65, 4953.2, 4393.55, 5257.7, 5534.025, 6173.825]):
    '''
        读取路网文件，并设置dijkstra法基础矩阵
    :param file_name:文件路径
            缺省vot方便用户自己选择是否输入
    :return:将原始存在直接路径的路阻进行赋值
    '''
    sf = shapefile.Reader(file_name, encoding='gbk')

    df = pd.DataFrame(columns=(
        'A', 'B', 'lw_time', 'C', 'V'))
    sf_records = sf.records()

    i = 1
    self_list = [0]
    for i in range(1, 1 + point_number):
        self_list.append(i * (point_number + 1))
    arr = np.arange(point_number ** 2)

    for i in range(point_number ** 2):
        arr[i] = 999999
        if i in self_list:
            arr[i] = 0
    dir_link = pd.DataFrame(arr.reshape((point_number, point_number)), columns=range(1, 1 + point_number),
                            index=(range(1, 1 + point_number)))

    for sf_record in sf_records:
        lw_time = (sf_record.Length / sf_record.Speed * 60 + 0.3 * vot[1])

        A = sf_record[3]
        B = sf_record[4]
        C = sf_record[9]
        dir_link[B][A] = lw_time
        dir_link[A][B] = lw_time
        df = df.append(pd.DataFrame({'A': [A], 'B': [B], 'lw_time': [lw_time], 'V': 0, 'C': [C]}))
    sf.close()

    return dir_link


def zdy_dijkstra(dir_link):
    '''
        dijkstra最短路径搜索
    :param dir_link:
    :return:各节点间最短路径路阻确定
    '''
    road_list = []
    point_number = len(dir_link)
    origin_dir_link = copy.deepcopy(dir_link)
    work_dir_link = copy.deepcopy(dir_link)
    for start in range(1, point_number + 1):  # 定义起始节点
        for end in range(1, point_number + 1):  # 定义目标节点
            link_list = []
            link_list.append(start)
            zdy_list = []  # 清空最短路径列表
            for k in range(1, point_number + 1):
                zdy_list.append(work_dir_link[start][k] + work_dir_link[k][end])  # 检索前节点到目标节点最短路

            if min(zdy_list) < work_dir_link[start][end]:
                work_dir_link[start][end] = min(zdy_list)  # 计算最短路与目标值对比
                work_dir_link[end][start] = work_dir_link[start][end]  # 矩阵对称同步
                link_list.append(zdy_list.index(min(zdy_list)) + 1)
            link_list.append(end)
            road_list.append(link_list)

    array1 = np.array(road_list).T
    df1 = pd.DataFrame(array1.reshape((point_number, point_number)), columns=range(1, 1 + point_number),
                      index=(range(1, 1 + point_number)))
    df1.to_csv("路径12213.csv", encoding="gbk", index=False)

    # 得到最短路径的路径list

    for l_list in road_list:
        i = 0
        # for i in range(len(l_list) - 1):
        print(l_list)
        while i < (len(l_list) - 1):
            if origin_dir_link[l_list[i]][l_list[i + 1]] == 999999 :
                print('cuowu',road_list[(l_list[i] - 1) * point_number + l_list[i + 1] - 1][1])
                l_list.insert(i + 1, road_list[(l_list[i] - 1) * point_number + l_list[i + 1] - 1][1])

            i+=1
    print('over')
    # 将路径list进行整列成为矩阵
    array = np.array(road_list).T
    # print(array)
    df = pd.DataFrame(array.reshape((point_number, point_number)), columns=range(1, 1 + point_number),
                      index=(range(1, 1 + point_number)))
    for i in range(1, point_number + 1):
        for j in range(i, point_number + 1):
            df[i][j] = list(reversed(df[j][i]))
    df.to_csv("路径.csv", encoding="gbk", index=False)
    return work_dir_link, df


def read_info(file_name, zonal_name):
    # 读取路网文件
    sf = shapefile.Reader(file_name)
    shapes = sf.shapes()
    fields = sf.fields
    records = sf.records()
    fields_name = []
    # 将路网信息从dbf转换为矩阵
    for field in fields:
        fields_name.append(field[0])
    del fields_name[0]
    table = []
    for record in records:
        table.append(record[:])
    net_info = pd.DataFrame(table, columns=fields_name)

    # 将点坐标与点建立对应矩阵
    point_dict = {}
    for i in range(len(shapes)):
        # for j in range(len(shapes[i].points)):
        point_dict[net_info['A'][i]] = shapes[i].points[0]
        point_dict[net_info['B'][i]] = shapes[i].points[1]
    point_info = pd.DataFrame(list(point_dict.items()), columns=['点', '坐标'])
    point_info.to_csv("点坐标信息.csv", encoding="gbk", index=False)

    # 读取小区数据
    zonal_sf = shapefile.Reader(zonal_name)
    zonal_shapes = zonal_sf.shapes()
    zonal_fields = zonal_sf.fields
    zonal_records = zonal_sf.records()

    # 小区周边邻接矩阵点读入
    zonal_point_info = {}

    for i in range(len(zonal_shapes)):
        zonal_point_list = []
        for j in range(point_info.shape[0]):
            if point_info['坐标'][j] in zonal_shapes[i].points:
                zonal_point_list.append(j)
        zonal_point_info['{}'.format(i)] = zonal_point_list
    return zonal_point_info


def zonal_inter_link(road_cost, zonal_point_info):
    # 计算区内出行时间：小区周边道路出行最短行程时间/2
    zonal_inter_cost = {}
    i = 1
    for value in zonal_point_info.values():

        inter_cost = []

        for origin in value:
            for end in value:
                if end != origin:
                    inter_cost.append(road_cost[origin + 1][end + 1])

        zonal_inter_cost[i] = min(inter_cost) / 2
        i += 1

    zonal_inter_cost_pd = pd.Series(zonal_inter_cost)

    zonal_dir_link_cost = pd.DataFrame(columns=range(1, len(zonal_point_info) + 1),
                                       index=range(1, len(zonal_point_info) + 1))

    zonal_point_info_pd = pd.Series(zonal_point_info)

    for i in range(len(zonal_point_info)):
        for j in range(len(zonal_point_info)):
            if i == j:
                zonal_dir_link_cost[i + 1][j + 1] = 0
            else:
                # 清空小区周边路网路阻列表
                zonal_cost = []
                # 拾取小区周边节点并进行基于对应节点进行阻抗计算
                for item_a in zonal_point_info_pd[i]:
                    for item_b in zonal_point_info_pd[j]:
                        if item_a == item_b:
                            continue
                        zonal_cost.append(road_cost[item_a + 1][item_b + 1])

                zonal_dir_link_cost[i + 1][j + 1] = min(zonal_cost)

    # 将计算所得小区区内出行时间加到路径出行时间上
    for i in range(1, 1 + len(zonal_point_info)):
        # print(zonal_inter_cost_pd[i])
        zonal_dir_link_cost[i] += zonal_inter_cost_pd[i]
        zonal_dir_link_cost[i][i] -= zonal_inter_cost_pd[i]
    return zonal_dir_link_cost


def gravity_calculation(traffic_cost, pa):
    '''
        计算各小区间交通量
    :param traffic_cost: 成本矩阵
    :param pa: PA矩阵
    :return:
    '''
    traffic_distribution = pd.DataFrame(columns=range(1, len(traffic_cost) + 1), index=range(1, len(traffic_cost) + 1))

    for i in range(len(traffic_cost)):
        for j in range(len(traffic_cost)):
            sum_list = []
            for k in range(len(traffic_cost)):
                # print(PA['A1'][k],traffic_cost.iloc[i][k+1])
                sum_list.append(pa['A1'][k] * traffic_cost.iloc[k][i + 1])

                # print(sum_list)
            # print(PA[1][i],PA[2][j],traffic_cost[i][j + 1],sum(sum_list))

            traffic_distribution.iloc[j][i + 1] = int(pa['P1'][i] * pa["A1"][j] * traffic_cost.iloc[j][i + 1] / sum(
                sum_list))
    traffic_distribution.insert(0, 'index', range(1, 1 + len(traffic_cost)))

    return traffic_distribution


def gravity_model(traffic_cost, PAmatrix):
    '''
        重力模型
    :param traffic_cost: 路阻函数
    :param PAmatrix: 产生吸引矩阵
    :return:
    '''
    # 试算调整gamma值
    gamma = 2
    # f（cij）函数运算
    traffic_cost = traffic_cost[traffic_cost != 0] ** -gamma
    traffic_cost = traffic_cost.fillna(0)
    traffic_distribution = gravity_calculation(traffic_cost, PAmatrix)
    return traffic_distribution


def bpr_caculation(t, vc, a=0.15, b=4):
    # 基于BPR函数计算加载交通量对路阻影响
    t = t * (1 + a * (vc ** b))
    return t


def assignment(file_name, zonal_name, road_cost, distribution_matrix, dir_link_df, link_time, zonal_point_info):
    zonal_point_info_pd = pd.Series(zonal_point_info)
    n_sf = shapefile.Reader(file_name)
    z_sf = shapefile.Reader(zonal_name)
    n_sf_records = n_sf.records()
    link_df = pd.DataFrame(columns=(
        'A', 'B', 'lw_time', 'c', 'v', 'opposite_v'))
    # 设置基础的路径矩阵
    for n_sf_record in n_sf_records:
        A = n_sf_record[3]
        B = n_sf_record[4]
        C = n_sf_record[9]
        lw_time = (n_sf_record.Length / n_sf_record.Speed * 60 + 0.3 * 3296.3)
        link_df = link_df.append(
            pd.DataFrame({'A': [A], 'B': [B], 'lw_time': [lw_time], 'v': 0, 'opposite_v': 0, 'c': [C]}))
        origin_link_df = copy.deepcopy(link_df)
    for col in range(0, len(zonal_point_info)):  # 检索每个交通小区的OD，这一步先不做，只做小区1 到小区2的路径
        for loc in range(1, len(zonal_point_info) + 1):
            number = 0
            if col + 1 != loc:  # 起终点小区不一致
                zonal_cost = {}
                # 拾取小区周边节点并进行确定小区之间最短阻抗的路径节点
                for item_a in zonal_point_info_pd[col]:
                    for item_b in zonal_point_info_pd[loc - 1]:
                        if item_a == item_b:
                            continue
                        zonal_cost[(item_a + 1, item_b + 1)] = road_cost[item_a + 1][item_b + 1]
                # print(zonal_cost)
                ori_zonal_min_dir = min(zonal_cost, key=zonal_cost.get)
                zonal_min_dir = ori_zonal_min_dir[:]

                # 以100为一个迭代单位采用容量限制增量分配法进行迭代分配
                while number < distribution_matrix.iloc[col, loc]:  # 判定一个小区分配完毕条件
                    work_list = dir_link_df[zonal_min_dir[0]][zonal_min_dir[1]]
                    for j in range(len(work_list) - 1):  # 针对最短路径中每条路进行分配流量
                        # print(col + 1, loc)
                        # print(work_list, j, distribution_matrix.iloc[col, loc] - number)
                        link_df.loc[((link_df.A == work_list[j]) & (link_df.B == work_list[j + 1])) | (
                                    (link_df.B == work_list[j]) & (link_df.A == work_list[j + 1])), 'v'] += 500
                        # 拾取正在工作的单位行
                        work_link = link_df.loc[((link_df.A == work_list[j]) & (link_df.B == work_list[j + 1])) | (
                                (link_df.B == work_list[j]) & (link_df.A == work_list[j + 1]))]

                        origin_work_link = origin_link_df.loc[
                            ((origin_link_df.A == work_list[j]) & (origin_link_df.B == work_list[j + 1])) | (
                                    (origin_link_df.B == work_list[j]) & (origin_link_df.A == work_list[j + 1]))]
                        print(work_list)
                        print(work_link)
                        print(origin_work_link)
                        new_lw_time = bpr_caculation(  # 计算分配交通量后的新路阻
                            float(origin_work_link.loc[0]['lw_time']),
                            (float(work_link.loc[0]['v']) / work_link.loc[0]['c']))
                        # 将新阻抗写入路径矩阵
                        link_df.loc[(link_df.A == work_list[j]) & (link_df.B == work_list[j + 1]) | (
                                (link_df.B == work_list[j]) & (
                                link_df.A == work_list[j + 1])), 'lw_time'] = new_lw_time
                        # 基于新路阻计算新的整个路网路阻及路阻函数
                        link_time[work_list[j]][work_list[j + 1]] = new_lw_time
                        link_time[work_list[j + 1]][work_list[j]] = new_lw_time
                        road_cost, dir_link_df = zdy_dijkstra(link_time)
                    number += 500

                else:  # 对于每次以100为迭代单位导致的剩余量进行相同操作
                    # print('该计算单元已结束')
                    for j in range(len(work_list) - 1):
                        link_df.loc[((link_df.A == work_list[j]) & (link_df.B == work_list[j + 1])) | (
                                (link_df.B == work_list[j]) & (link_df.A == work_list[j + 1])), 'v'] += \
                            distribution_matrix.iloc[col, loc] + 500 - number
                        work_link = link_df.loc[((link_df.A == work_list[j]) & (link_df.B == work_list[j + 1])) | (
                                (link_df.B == work_list[j]) & (link_df.A == work_list[j + 1]))]

                        origin_work_link = origin_link_df.loc[
                            ((origin_link_df.A == work_list[j]) & (origin_link_df.B == work_list[j + 1])) | (
                                    (origin_link_df.B == work_list[j]) & (origin_link_df.A == work_list[j + 1]))]
                        new_lw_time = bpr_caculation(
                            float(origin_work_link.loc[0]['lw_time']),
                            (float(work_link.loc[0]['v']) / work_link.loc[0]['c']))
                        # 计算新阻抗
                        link_df.loc[(link_df.A == work_list[j]) & (link_df.B == work_list[j + 1]) | (
                                (link_df.B == work_list[j]) & (
                                link_df.A == work_list[j + 1])), 'lw_time'] = new_lw_time

                        link_time[work_list[j]][work_list[j + 1]] = new_lw_time
                        link_time[work_list[j + 1]][work_list[j]] = new_lw_time
                        road_cost, dir_link_df = zdy_dijkstra(link_time)
        # link_df为携带着加载交通量的路网信息文件
    print(link_df)
    return road_cost, dir_link_df, link_df


def main():
    file_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\路网.dbf'
    zonal_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
    link_time = read_shp(file_name, 12)

    origin_dir_link = copy.deepcopy(link_time)  # 对于读取文件获得的路网基础路阻文件进行备份
    road_cost, dir_link_df = zdy_dijkstra(link_time)  # 采用dijkstra算法得到路网路阻及对应的最短路径文件

    zonal_point_info = read_info(file_name, zonal_name)  # 读取小区文件中，每个小区周边的节点信息

    for i in range(3):
        traffic_cost = zonal_inter_link(road_cost, zonal_point_info)  # 计算小区之间的出行成本
        traffic_cost.to_csv("出行成本.csv", encoding="gbk", index=False)
        PAfile_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行生成\\预测PA矩阵.csv'
        PAfile_data = pd.read_csv(PAfile_name, low_memory=False)  # 防止弹出警告
        PAfile_df = pd.DataFrame(PAfile_data)

        distribution_matrix = gravity_model(traffic_cost, PAfile_df)

        # 返回分配后的加载路阻，对应各节点之间最短路径，携带加载流量的路径文件
        road_cost, dir_link_df, link_df = assignment(file_name, zonal_name, road_cost, distribution_matrix, dir_link_df,
                                                     origin_dir_link, zonal_point_info)

    distribution_matrix.to_csv("分布矩阵.csv", encoding="gbk", index=False)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # distribution_matrix[1:].plot(kind='bar', stacked=True, title='各小区间分布交通量', edgecolor='black', linewidth=1, alpha=0.8,
    #                              rot=0)
    # geopandas用于绘图
    # nybb_path = gdf.datasets.get_path('交通小区')

    # # 可视化部分代码
    # nybb_path = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
    # n = gdf.GeoDataFrame.from_file(file_name, encoding='gb18030')
    # g = gdf.GeoDataFrame.from_file(nybb_path, encoding='gb18030')
    # # g['geometry'].convex_hull
    # # # g.set_index('BoroCode', inplace=True)
    # # # g.sort()
    # g['centroid_column'] = g.centroid
    # # g.coords[:]
    # # print(type(g['centroid_column'][1]))
    # # print(g['centroid_column'][1])
    # zuobiao = []
    #
    # for i in range(4):
    #     zuobiao.append([g['centroid_column'][i].x, g['centroid_column'][i].y])
    # # print(zuobiao)
    # # plt.plot(zuobiao[1], zuobiao[2], color='r')
    # # for i in range(4):
    # #     for j in range(4):
    # #         if i != j :
    # #             plt.plot(zuobiao[i],zuobiao[j], color='r')
    # #             plt.scatter(zuobiao[i],zuobiao[j], color='b')
    # # g.plot()
    # t = g.set_geometry('centroid_column')
    # # print(t)
    # # print(type(g))
    # # print(g['centroid_column'])
    # # for i in range(4):
    # #     for j in range(4):
    # #         if i != j:
    # #             print(g['centroid_column'][i], g['centroid_column'][j])
    # # g.plot(g['centroid_column'][i], g['centroid_column'][j], color='r')
    # # 出图部分
    # ax1 = g.plot(color='blue')
    # ax2 = n.plot(color='black', ax=ax1)
    # t.plot(ax=ax2, color='r')
    # #
    # # # print(g['geometry'])
    # plt.show()
    # # # print(g)


if __name__ == '__main__':
    time1 = time.time()
    main()
    time2 = time.time()
    # 输出执行时间
    print(f'耗时：{time2 - time1}秒')
    # #打开文件
    # os.system('分布矩阵.csv')
