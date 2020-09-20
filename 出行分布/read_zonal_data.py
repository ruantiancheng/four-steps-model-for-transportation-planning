import numpy as np
import pandas as pd
import dbfread as dbf
import csv
import shapefile

file_name = 'C:\\Users\\Administrator\\Desktop\\python_work\\毕业设计\\出行分布\\示范路网小区\\交通小区.dbf'
sf = shapefile.Reader(file_name)
shapes = sf.shapes()
print(len(shapes.points))
point_list = {}
for i in range(len(shapes)):
    for j in range(len(shapes[i].points)):
        pass