#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 10.07.20 14:33
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH0_4_PyTorch2NumPy.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import torch
import numpy as np

# PyTorch --> NumPy
points = torch.randn(3, 4)
points_np =points.numpy()
print(points)
print(points_np)

print("----------------------------------------")

# NumPy --> PyTorch
points_fromnp = torch.from_numpy(points_np)
print(points_fromnp)

