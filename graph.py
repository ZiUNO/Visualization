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


def data_verify(origin_data):
    keys = origin_data.keys()
    for s in origin_data:
        if origin_data[s] is None:
            continue
        for e in origin_data[s]:
            if e not in keys:
                raise KeyError('终边在图中未出现')
            if origin_data[e] is None:
                continue
            tmp_keys = origin_data[e].keys()
            if s not in tmp_keys:
                continue
            if origin_data[s][e] == origin_data[e][s]:
                del origin_data[e][s]
            else:
                raise ValueError('边值不相等：%d->%d & %d->%d' % (s, e, e, s))
    return origin_data


def format_transform(verified_data):
    size = len(verified_data)
    s = []
    e = []
    out = []
    for start in verified_data:
        if verified_data[start] is None:
            continue
        for end in verified_data[start]:
            if verified_data[start][end] is None:
                continue
            s.append(start)
            e.append(end)
            out.append(verified_data[start][end])
    for index in range(len(s)):
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
    size = len(s[0])
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
    xx, yy, loss = sess.run((x, y, loss), feed_dict={start: s, end: e, length: out})
    points = []
    xx = [list(tmp)[0] for tmp in xx]
    yy = [list(tmp)[0] for tmp in yy]
    if len(xx) != len(yy):
        raise RuntimeError
    for index in range(len(xx)):
        points.append([xx[index], yy[index]])
    sess.close()
    return points, loss
