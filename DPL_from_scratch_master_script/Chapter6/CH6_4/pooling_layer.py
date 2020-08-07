#!/usr/bin/env python
# coding=utf-8
# '''
# @Author       : Jingsheng Lyu
# @Date         : 2020-07-09 22:52:39
# @LastEditors  : Jingsheng Lyu
# @LastEditTime : 2020-07-09 23:05:10
# @FilePath     : /Deep_Learning/Chapter6/CH6_4/pooling_layer.py
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com
# '''

import numpy as np
from util import *

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):


        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展开 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # 最大值 (2)
        out = np.max(col, axis=1)
        # 转换 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out
