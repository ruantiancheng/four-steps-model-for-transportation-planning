'''
    出行生成模块
    输入：file_name：人口及预测数据；PA计算回归公式
    输出：预测PA矩阵
'''

import numpy as np
import pandas as pd
import dbfread as dbf
import csv
import xlwt
import datetime
import os


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
        df = pd.DataFrame(columns=(
            'index', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
            'A7',
            'A8', 'A9', 'A10'))

        for i in range(50):
            P1 = int(23.968 * table.records[i]['TEMP_POP'] + 13.492 * table.records[i]['NAGR_STU'] - 55647.229 *
                     table.records[i]['DUIS_33'] + 234682.728 * table.records[i]['CYZQCS_1'] - 33178.606 *
                     table.records[i][
                         'DAOS_42'] + 10549.388 * table.records[i]['DSB_412'] + 0.654 * table.records[i]['TECHNICAL'])
            P2 = int(3398.607 * table.records[i]['FZX_1'] + 0.091 * table.records[i]['TECHNICAL'] + 558.268)
            P3 = int(
                31.441 * table.records[i]['NCA_NAGR_P'] - 1.654 * table.records[i]['CA_POP'] + 4.122 * table.records[i][
                    'YDS_25'] + 5.122 * table.records[i]['YDS_33'] + 33.095 * table.records[i]['TOTAL_STU'] + 35.636 * \
                table.records[i]['DUIS_41'] + 64.763 * table.records[i]['DAOS_42'] - 5167.338 * table.records[i][
                    'DSH_412'] + 5.447 * table.records[i]['C_AGR'] + 0.275 * table.records[i]['TECHNICAL'] - 2.57 * \
                table.records[i]['VCS'])
            P4 = int(3.516 * table.records[i]['TEMP_POP'] - 11.318 * table.records[i]['TEMT_STU'])
            P5 = int(
                38.911 * table.records[i]['DSB_12'] - 792.465 * table.records[i]['PFGFZX_5'] - 76.852 *
                table.records[i][
                    'DSB_56'] + 7.581 * table.records[i]['DAOS_7'] + 1.806 * table.records[i]['DSB_212'] + 6.396 *
                table.records[i]['CYZGZRK_3'] + 81 * table.records[i]['CA_WORKER'] - 26.923 * table.records[i][
                    'NCA_AGR_ST'] + 2.209 * table.records[i]['CA_STU'] - 16.317 * table.records[i]['YDS_34'] - 84.809 *
                table.records[i]['DAOS_34'] - 3.2 * table.records[i]['PRIV_CAR'] + 4.192 * table.records[i][
                    'CYZQCS_1'] + 4.536 * table.records[i]['YDS_41'] - 0.234 * table.records[i]['UNIT_CAR'] - 19.737 *
                table.records[i]['DAOS_42'] - 304.115 * table.records[i]['DSH_412'] - 121.211 * table.records[i][
                    'DSB_412'] - 0.101 * table.records[i]['MANU'] + 1.31 * table.records[i]['C_AGR'] + 0.045 *
                table.records[i]['TECHNICAL'] - 1014.87)
            P6 = int(
                -1346.7 * table.records[i]['PFGFZX_1'] + 359.369 * table.records[i]['DUIS_5'] + 212.124 *
                table.records[i][
                    'DAOS_41'] + 1.814 * table.records[i]['TECHNICAL'] + 423.01)
            P7 = int(831.307 * table.records[i]['CA_AGR_STU'] + 1568.597 * table.records[i]['DSB_412'] - 0.455 * \
                     table.records[i]['COMMERCIAL'] + 6.065 * table.records[i]['CONSTRUC'] - 0.579 * table.records[i][
                         'PRIMARY'])
            P8 = int(1406.788 * table.records[i]['CA_AGR_STU'] + 0.548 * table.records[i]['TECHNICAL'])
            P9 = int(-17.843 * table.records[i]['NCA_AGR_PO'] + 15.152 * table.records[i]['CA_STU'])
            P10 = int(
                151.483 * table.records[i]['DAOS_2'] - 902.88 * table.records[i]['DSB_12'] - 295.873 * table.records[i][
                    'PFGFZX_5'] + 902.831 * table.records[i]['DAOS_7'] + 637.124 * table.records[i][
                    'DSB_212'] + 156.757 * \
                table.records[i]['CYZGZRK_3'] - 2202.53 * table.records[i]['DSB_234'] + 2.696 * table.records[i][
                    'CA_WORKER'] + 0.115 * table.records[i]['TECHNICAL'] - 3428.282)
            A1 = int(
                15.655 * table.records[i]['TEMP_POP'] + 0.508 * table.records[i]['TECHNICAL'] + 15600.2 *
                table.records[i][
                    'DSB_412'])
            A2 = int(0.091 * table.records[i]['TECHNICAL'] + 5550.183 * table.records[i]['DSB_412'])
            A3 = int(
                2.811 * table.records[i]['TEMP_POP'] + 1540.784 * table.records[i]['DUIS_31'] - 20.186 *
                table.records[i][
                    'TEMT_STU'] + 0.14 * table.records[i]['TECHNICAL'] + 1120.081 * table.records[i]['DSB_412'])
            A4 = int(-5620.836 * table.records[i]['DUIS_32'] + 1.779 * table.records[i]['NAGR_STU'] + 1344.195 * \
                     table.records[i]['DSB_412'] - 0.405 * table.records[i]['C_SERVICE'] + 0.162 * table.records[i][
                         'TECHNICAL'])
            A5 = int(
                2.21 * table.records[i]['TEMP_POP'] + 0.335 * table.records[i]['CA_STU'] - 0.151 * table.records[i][
                    'C_SERVICE'] + 0.083 * table.records[i]['TECHNICAL'])
            A6 = int(-40.501 * table.records[i]['NCA_AGR_PO'] - 41046.051 * table.records[i]['PFGFZX_1'] + 42.289 * \
                     table.records[i]['CA_STU'] - 37721.026 * table.records[i]['YDS_42'] + 136.153 * table.records[i][
                         'DSB_412'] - 3.974 * table.records[i]['C_SERVICE'] + 1.489 * table.records[i][
                         'TECHNICAL'] + 802022.399)
            A7 = int(
                4.051 * table.records[i]['TEMP_POP'] + 5331.557 * table.records[i]['DAOS_31'] + 476.853 *
                table.records[i][
                    'CA_AGR_STU'] - 26.84 * table.records[i]['TEMT_STU'] + 3.375 * table.records[i][
                    'CA_STU'] + 50234.853 * \
                table.records[i]['DUIS_41'] - 1932.56 * table.records[i]['DAOS_42'] + 3610.825 * table.records[i][
                    'DSB_412'] - 0.224 * table.records[i]['COMMERCIAL'] + 0.182 * table.records[i]['TECHNICAL'])
            A8 = int(7.381 * table.records[i]['TEMP_POP'] + 915.111 * table.records[i]['CA_AGR_STU'] - 52.927 * \
                     table.records[i]['TEMT_STU'] - 10446.220 * table.records[i]['YDS_42'] + 4837.942 *
                     table.records[i][
                         'DSB_412'] + 0.253 * table.records[i]['TECHNICAL'] + 100002.545)
            A9 = int(9.476 * table.records[i]['NCA_AGR_PO'] + 9578.779 * table.records[i]['YDS_42'] - 4190.893 *
                     table.records[i]['DAOS_42'] + 4978.961 * table.records[i]['DSB_412'] - 1.075 * table.records[i][
                         'C_SERVICE'] - 0.467 * table.records[i]['COMMERCIAL'] + 8.122 * table.records[i][
                         'CONSTRUC'] + 0.356 * table.records[i]['TECHNICAL'] - 0.605 * table.records[i]['SECONDARY'])
            A10 = int(
                -5.03 * table.records[i]['NCA_AGR_PO'] + 47.976 * table.records[i]['DAOS_31'] + 5.863 *
                table.records[i][
                    'CA_STU'] - 0.593 * table.records[i]['C_SERVICE'] + 0.209 * table.records[i]['TECHNICAL'])
            df = df.append(pd.DataFrame(
                {'index': [i + 1], 'P1': [P1], 'P2': [P2], 'P3': [P3], 'P4': [P4], 'P5': [P5], 'P6': [P6], 'P7': [P7],
                 'P8': [P8], 'P9': [P9], 'P10': [P10], 'A1': [A1], 'A2': [A2], 'A3': [A3], 'A4': [A4], 'A5': [A5],
                 'A6': [A6], 'A7': [A7],
                 'A8': [A8], 'A9': [A9], 'A10': [A10]}), ignore_index=True)


        #p2a 过程
        rate = pd.DataFrame(columns=('A1', 'A2', 'A3', 'A4', 'A5', 'A6','A7','A8', 'A9', 'A10'))
        productsum = []
        for i in df.columns:
            # print(df[i].sum())
            productsum.append(df[i].sum())
            # print(product)
        for j in range(11,21):
            origin_rate = []
            for i in range(50):
                origin_rate.append(df.iloc[i,j] / productsum[j])
                # print(sum(origin_rate))
            # print(origin_rate)
            # rate = rate.append(pd.DataFrame([{'A{}'.format(j-10):origin_rate}]), ignore_index=True,sort=False)
            rate['A{}'.format(j-10)] = origin_rate

        # print(rate)

        for i in range(1, 11):
            if productsum[i] != productsum[i + 10]:
                # for j in range(50):\\\\\\\\
                    df['A{}'.format(i)] = productsum[i] * rate['A{}'.format(i)]



        # print(product)
        return df


def main():
    file_name = '总表-变换数据-第三组_2020年预测用数据.dbf'
    generation = Generation(file_name)
    table = generation.read_file()
    df = generation.calculation_model(table)
    df.to_csv("预测PA矩阵.csv", encoding="gbk", index=False)  # 不传递index


if __name__ == '__main__':
    main()
