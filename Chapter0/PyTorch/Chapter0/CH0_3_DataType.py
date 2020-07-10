#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 10.07.20 14:01
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH0_3_DataType.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import torch

# Data Type for PyTorch
# float, int, double, FatTensor, DoubleTensor and so on

double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1,2], [3,4]], dtype=torch.short)
print(short_points)
print(double_points)

print("-----------------------------------")

# Transformation of Data Type
zeros = torch.zeros(10, 2)
print(zeros.dtype)

# Two ways to transform the data type
double_zeros = zeros.double()
print(double_zeros)

to_double_zeros = zeros.to(dtype=torch.double)
print(to_double_zeros)

print("-----------------------------------")

print("Index for PyTorch")
randn_points = torch.randn([3,4], dtype=torch.double)
print(randn_points)
print(randn_points[1:])
print(randn_points[1:, :2])
print(randn_points[1:, 0])

