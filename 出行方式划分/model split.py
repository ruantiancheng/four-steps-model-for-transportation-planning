# -*- coding: utf-8 -*-
'''
    输入：分布交通量矩阵
        路网gis.shp文件
    输出：各方式划分交通量
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


def logit(dataframe):
    #logit 模型算法
    split_choice = []
    for item in dataframe.items:
        sum_list = []
        for item in dataframe.items:
            real_resist = math.e ** item
            sum_list.append(real_resist)
        split_choice.append( math.e ** item / sum(sum_list))
    return split_choice

def main():
    netshape_path = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\路网.dbf'
    zonal_shape_path = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
    nsf = shapefile.Reader(netshape_path, encoding='gbk')
    zsf = shapefile.Reader(zonal_shape_path, encoding='gbk')

    split_choice = logit(nsf)



if __name__ == '__main__':
    time1  = time.time()
    main()
    time2 = time.time()
    #输出执行时间
    print(f'耗时：{time2 - time1}秒')
