# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/12/17 9:22
"""
from graph import data_transform, tf_get_points

if __name__ == '__main__':
    data = {
        0:
            {1: 0.3,
             2: 0.3},
        1:
            {2: 0.5,
             3: 0.9}}
    data = data_transform(data)
    total_points = tf_get_points(data)
    for tmp in total_points:
        print(tmp)
