#!/usr/bin/env python
# coding=utf-8
# '''
# @Author       : Jingsheng Lyu
# @Date         : 2020-07-09 22:52:25
# @LastEditors  : Jingsheng Lyu
# @LastEditTime : 2020-07-09 22:55:40
# @FilePath     : /Deep_Learning/Chapter6/CH6_4/cnn_layer.py
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com
# '''
import numpy as np
from util import *

from Chapter6.CH6_4.util import im2col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 滤波器的展开
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out