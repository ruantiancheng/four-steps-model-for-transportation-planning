import numpy as np
import pandas as pd
import dbfread as dbf
import csv
import shapefile

file_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\路网.dbf'
def read_info(file_name):
    sf = shapefile.Reader(file_name)
    shapes = sf.shapes()
    fields = sf.fields
    records = sf.records()
    fields_name = []
    for field in fields:
        fields_name.append(field[0])
    del fields_name[0]
    table = []
    for record in records:
        table.append(record[:])
    net_info = pd.DataFrame(table, columns=fields_name)
    point_dict = {}
    for i in range(len(shapes)):
        # for j in range(len(shapes[i].points)):
        point_dict[net_info['A'][i]] = shapes[i].points[0]
        point_dict[net_info['B'][i]] = shapes[i].points[1]
    point_info = pd.DataFrame(list(point_dict.items()), columns=['点','坐标'])
    print(point_info)
    net_info.to_csv("{}信息.csv".format(file_name), encoding="gbk", index=False)
    return net_info
net_info = read_info(file_name)


# def Shp2dataframe(file_name):
#     '''将arcpy表单变为pandas表单输出'''
#     sf = shapefile.Reader(file_name)
#     fields = sf.shapeRecords()
#     print(len(fields))
#     table = []
#     fieldname = [field.record[0] for field in fields]
#     # 游标集合，用for 循环一次后没办法循环第二次!一个游标实例只能循环一次
#     data = shapefile.Reader(file_name)
#     for row in data:
#         # Shape字段中的要数是一个几何类
#         r = []
#         for field in fields:
#             r.append(row.fieldname)
#         table.append(r)
#     return pd.DataFrame(table, columns=fieldname)
# df = Shp2dataframe(file_name)
# print(df)
