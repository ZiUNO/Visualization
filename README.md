# Visualization
## graph
* 无向图的数据格式转换和二维点坐标生成
###### data_transform(origin_data)
* 传入无向图（不能出现重复边（例：0->1和1->0不能同时出现））
* 输入格式:
   `{0:
        {1: w01,
         2: w02,
         ...},
     1:
        {2: w12,
         ...},
     ...}`
###### tf_get_points(data, train_times=500)
* 输入格式:
    data_transform的输出
* 输出格式:
    `[[x1, y1], [x2, y2], ...]`
## Version
    Python 3.6.5
    Tensorflow 1.12.0
