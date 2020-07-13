#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 13.07.20 08:58
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH1_2_TimeSeries.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import torch
import numpy as np

data_path = ".../hour-fixed.csv"
bikes_numpy = np.loadtxt(data_path, dtype=np.float32, delimiter=",", skiprow=1, converters={1:lambda x:float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
print(bikes)

print(bikes.shape, bikes.stride())

# Return a new tensor using .view(), this function can change
# the dimension and the stride, but don't change the storage.
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
# daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())



