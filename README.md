# Visualization
## graph
* 无向图的数据格式转换和二维点坐标生成
###### data_verify(origin_data)
* 对原始数据进行校验
* 输入格式:
    * 可出现重复边（例：0->1和1->0若同时出现值需相等，否则报错）
    * 只有入度但没有出度的顶点标号需出现在图中（内容需置为None）
    * E.g.: 
        `{0:
            {1: w01,
             2: w02,
             ...},
          1:
            {2: w12,
             ...},
          ...}`
* 输出: 校验后的数据
###### format_transform(verified_data)
* 输入: data_verify校验后的数据
* 输出: 可用于训练的格式的数据
###### tf_get_points(data, train_times=500)
* 输入格式: format_transform的输出（训练次数默认500次）
* 输出格式: 
    * points, loss
    * E.g.: `[[x0, y0], [x1, y1], [x2, y2], ...], loss`
    * 每个元素表示一个坐标值`[x, y]`
    * loss表示当前数据误差值
## Version
    Python 3.6.5
    Tensorflow 1.12.0
