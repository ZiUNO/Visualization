# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/12/17 8:17
"""

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def data_transform(origin_data):
    s = []
    e = []
    out = []
    for start in origin_data:
        for end in origin_data[start]:
            s.append(start)
            e.append(end)
            out.append(origin_data[start][end])
    size = len(out)
    for index in range(size):
        tmp_s = np.zeros([size], dtype=np.float32)
        tmp_e = np.zeros([size], dtype=np.float32)
        start_ = s[index]
        end_ = e[index]
        tmp_s[start_] = 1
        tmp_e[end_] = 1
        s[index] = tmp_s
        e[index] = tmp_e
    out = [[tmp] for tmp in out]
    return [s, e, out]


def tf_get_points(data, train_times=500):
    [s, e, out] = data
    size = len(out)
    start = tf.placeholder(tf.float32, shape=[None, size])
    end = tf.placeholder(tf.float32, shape=[None, size])
    length = tf.placeholder(tf.float32, shape=[None, 1])
    x = tf.Variable(tf.random_normal([size, 1]))
    y = tf.Variable(tf.random_normal([size, 1]))
    x1 = tf.matmul(start, x)
    x2 = tf.matmul(end, x)
    y1 = tf.matmul(start, y)
    y2 = tf.matmul(end, y)
    x_ = tf.squared_difference(x1, x2)
    y_ = tf.squared_difference(y1, y2)
    outputs = tf.sqrt(tf.add(x_, y_))
    loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(outputs, length)))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(train_times):
        sess.run(train_step, feed_dict={start: s, end: e, length: out})
    xx, yy = sess.run((x, y), feed_dict={start: s, end: e, length: out})
    points = []
    xx = [list(tmp)[0] for tmp in xx]
    yy = [list(tmp)[0] for tmp in yy]
    if len(xx) != len(yy):
        raise RuntimeError
    for index in range(len(xx)):
        points.append([xx[index], yy[index]])
    sess.close()
    return points
