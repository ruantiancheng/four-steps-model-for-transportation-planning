import pandas as pd
import matplotlib.pyplot as plt
from sympy import *
import geopandas as gdf
#底图绘制
nybb_path = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
g = gdf.GeoDataFrame.from_file(nybb_path, encoding='gb18030')

nybb2_path = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\路网.dbf'
r = gdf.GeoDataFrame.from_file(nybb2_path, encoding='gb18030')

ax1 = g.plot(color='w')
r.plot(ax=ax1,color='red')


file_path1 = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\可视化出图\\数据.csv'
df1 = pd.read_csv(file_path1, low_memory=False)  # 防止弹出警告
file_path2 = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\分布矩阵.csv'
df2 = pd.read_csv(file_path2, low_memory=False)  # 防止弹出警告
df3 = pd.DataFrame(df1['centroid_column'])
file_path = 'C:\\Users\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\点坐标信息.csv'
df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')  # 防止弹出警告

x_coor = []
y_coor = []
for g in range(6):
    coor_list = df3.iloc[g, 0].split(' ')
    a_coor = eval(coor_list[1][1:])
    b_coor = eval(coor_list[2][:-1])
    x_coor.append(a_coor)
    y_coor.append(b_coor)


zonal_point_info = {0: [1, 2, 4, 5], 1: [4, 5, 7, 8], 2: [2, 3, 5, 6], 3: [5, 6, 8, 9], 4: [7, 8, 10, 11],
                    5: [8, 9, 11, 12]}
for g in range(6):
    px_list = []
    py_list = []
    for i in zonal_point_info[g]:
        #     print(type(df.iloc[i-1][1]))
        zonal_coor_list = df.iloc[i - 1][1].split(',')
        #     print(zonal_coor_list)
        px_list.append(eval(zonal_coor_list[0][1:]))
        py_list.append(eval(zonal_coor_list[1][1:-1]))
    plt.plot(px_list, py_list, "ro", linewidth=1, color='gray')

    for k in range(len(px_list)):
        draw_x_coor = [px_list[k], x_coor[g]]
        draw_y_coor = [py_list[k], y_coor[g]]
        plt.plot(draw_x_coor, draw_y_coor, linewidth=1, color='gray')

plt.plot(x_coor, y_coor,"ro",linewidth=1, color='yellow')


file_path = 'C:\\Users\Administrator\\Desktop\\python_work\\毕业设计\\可视化出图\\路段流量表.csv'
df = pd.read_csv(file_path, low_memory = False)#防止弹出警告

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
color_choice = 'y'
for i in range(len(df)):
    x_coor = []
    y_coor = []
    a_coor = eval(df.iloc[i, 2])
    b_coor = eval(df.iloc[i, 4])
    print(type(a_coor[0]))
    volumn = df.iloc[i, 8]
    lw_choice = 12 * volumn / 7200
    if volumn < 1400:
        color_choice = 'g'
        lw_choice = 1
    elif 1400 <= volumn < 3600:
        color_choice = '#ADFF2F'
    elif 3600 <= volumn < 5760:
        color_choice = '#FFA500'
    else:
        color_choice = 'r'
    print(a_coor[0])
    x_coor.append(a_coor[0])
    x_coor.append(b_coor[0])
    y_coor.append(a_coor[1])
    y_coor.append(b_coor[1])

    plt.plot(x_coor, y_coor, linewidth=lw_choice, color=color_choice)
    # 出vc比图

    plt.text((a_coor[0] + b_coor[0]) / 2, (a_coor[1] + b_coor[1]) / 2, s='{:.2%}'.format(volumn / 7200), ha='center',
             va='bottom', fontsize=20)
    plt.title('路段流量图',fontsize = 20)
    #出流量图
    # plt.text((a_coor[0]+b_coor[0])/2,(a_coor[1]+b_coor[1])/2,s=volumn, ha='center', va= 'bottom',fontsize=20)


    plt.axis('off')
plt.show()