# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/12/17 9:22
"""
from graph import *

if __name__ == '__main__':
    data = {
        0:
            {1: 0.3,
             2: 0.3},
        1:
            {2: 0.5,
             3: 0.9},
        2:
            {1: 0.5,
             3: 0.8,
             4: 0.1,
             5: 0.7},
        3:  None,
        4:  None,
        5:  None,
        6:
            {0: 0.9}}
    data = data_verify(data)
    data = format_transform(data)
    total_points, loss = tf_get_points(data)
    for tmp in total_points:
        print(tmp)
    print(loss)
