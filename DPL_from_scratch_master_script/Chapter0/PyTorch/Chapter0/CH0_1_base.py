#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 08.07.20 18:50
# @Author       : Jingsheng Lyu
# @Site         :
# @File         : CH0_2_Derivative.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

# Verify the install of PyTorcch
import torch
print(torch.__version__)

print("-----------------------------")

# Create a new random tensor with (2, 3) dim
x = torch.randn(2, 3)
print(x)

print("-----------------------------")

# Returns a tensor filled with uninitialized data
empty = torch.empty(2, 3)
print(empty)

print("-----------------------------")

# Create a new 2-dim tensor(matrix) with all
zeros = torch.zeros(2, 3, dtype=torch.long)
print(zeros)

print("-----------------------------")

# Create a new matrix with all 1
new_matrix = x.new_ones(5, 3, dtype=torch.double)
print(new_matrix)

print("-----------------------------")

# Input the data
new_data = torch.tensor([5.5, 3.])
print(new_data)

print("-----------------------------")

# Create a new similar matrix
similar_matrix = torch.randn_like(x, dtype=float)
print(similar_matrix)

print("-----------------------------")

# Show the size
print(x.size())

print("-----------------------------")

# Reshape in NumPy or resize in PyTorch
rd = torch.randn(4, 4)
r_rd = rd.view(16)
rr_rd = rd.view(-1, 8)
print(rd.size(), r_rd.size(), rr_rd.size())

print("-----------------------------")

# PyTorch <--> NumPy
print("PyTorch -> NumPy")
import numpy as np
torch_ones_1 = torch.ones(5)
np_ones_1 = torch_ones_1.numpy()
print(np_ones_1)

print("-----------------------------")

print("NumPy -> PyTorch")
np_ones_2 = np.ones(4)
torch_ones_2 = torch.from_numpy(np_ones_2)
print(torch_ones_2)
