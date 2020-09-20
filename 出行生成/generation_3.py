'''
    出行生成模块
    输入：file_name：人口及预测数据；PA计算回归公式
    输出：预测PA矩阵
    追加功能：对p2a ,a2p功能进行封装
'''

import numpy as np
import pandas as pd
import dbfread as dbf
import csv
import xlwt
import datetime
import os
import shapefile
import time



class Generation:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_file(self):
        '''
            读取输入文件
        '''
        table = dbf.DBF(self.file_name, load=True)
        return table

    def calculation_model(self, table):
        # 回归所得pa矩阵输入
        df = pd.DataFrame(columns=(
            'index', 'P1', 'A1'))

        for i in range(len(table)):
            P1 = int(0.002*table.records[i]['Area']+0.3*table.records[i]['Stu'])
            A1 = int(0.005*table.records[i]['Area']-0.0003*table.records[i]['Pop'])
            df = df.append(pd.DataFrame(
                {'index': [i + 1], 'P1': [P1], 'A1': [A1]}))
        return df

    def p2a(self, df):
        # p2a 过程
        behave_number = int((df.shape[1]-1)/2)
        for i in range(behave_number):
            rate = pd.DataFrame(columns=['A{}rate'.format(i+behave_number)])
        productsum = []
        for i in df.columns:
            productsum.append(df[i].sum())
        for j in range(1+behave_number,1+behave_number*2):
            origin_rate = []
            for i in range(len(df)):
                origin_rate.append(df.iloc[i, j] / productsum[j])
            rate['A{}rate'.format(j-1 )] = origin_rate
        actually_attractive = []
        for i in range(behave_number,behave_number*2):
            if productsum[i] != productsum[i + behave_number]:
                for j in range(df.shape[0]):
                    actually_attractive.append(int(productsum[i] * rate['A{}rate'.format(i)][j]))
                df['A{}'.format(i)] =actually_attractive
        return df

    def a2p(self, df):
        # a2p 过程
        rate = pd.DataFrame(columns=('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'))
        attractive_sum = []
        for i in df.columns:
            # print(df[i].sum())
            attractive_sum.append(df[i].sum())
            # print(product)
        for j in range(11):
            origin_rate = []
            for i in range(50):
                origin_rate.append(df.iloc[i, j] / attractive_sum[j])
                # print(sum(origin_rate))
            # print(origin_rate)
            # rate = rate.append(pd.DataFrame([{'A{}'.format(j-10):origin_rate}]), ignore_index=True,sort=False)
            rate['P{}'.format(j)] = origin_rate

        for i in range(1, 11):
            if attractive_sum[i] != attractive_sum[i + 10]:
                # for j in range(50):\\\\\\\\
                df['P{}'.format(i)] = attractive_sum[i] * rate['P{}'.format(i)]
        # print(product)
        return df

    def nml(self, df):
        # nml 过程
        rate = pd.DataFrame(columns=('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'))
        attractive_sum = []
        for i in df.columns:
            # print(df[i].sum())
            attractive_sum.append(df[i].sum())
            # print(product)
        for j in range(11):
            origin_rate = []
            for i in range(50):
                origin_rate.append(df.iloc[i, j] / attractive_sum[j])
                # print(sum(origin_rate))
            # print(origin_rate)
            # rate = rate.append(pd.DataFrame([{'A{}'.format(j-10):origin_rate}]), ignore_index=True,sort=False)
            rate['P{}'.format(j)] = origin_rate

        for i in range(1, 11):
            if attractive_sum[i] != attractive_sum[i + 10]:
                # for j in range(50):\\\\\\\\
                df['P{}'.format(i)] = attractive_sum[i] * rate['P{}'.format(i)]
        # print(product)
        return df


def main():
    file_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
    generation = Generation(file_name)
    table = generation.read_file()
    df = generation.calculation_model(table)
    balance_df = generation.p2a(df)
    # print(df)
    # print(balance_df)
    balance_df.to_csv("预测PA矩阵.csv", encoding="gbk", index=False)  # 不传递index


if __name__ == '__main__':
    time1  = time.time()
    main()
    time2 = time.time()
    print(f'耗时：{time2 - time1}秒')
    os.system('预测PA矩阵.csv')
