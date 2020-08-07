#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 12.07.20 09:47
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH1_1_TableData.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import csv
import numpy as np
import torch

# With NumPy to load the txt
wine_path = "/home/jingsheng/Deep_Learning/Chapter0/PyTorch/Chapter1/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
# Read the data without the first row,because it is a string of the column
print(wineq_numpy)

print("----------------------------------------------")

# Verify if we have already read all data.
col_list = next(csv.reader(open(wine_path), delimiter=";"))
print(wineq_numpy.shape, col_list)

print("----------------------------------------------")

# Transform the data from NumPy to PyTorch
wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape, wineq.type())

# There are here 3 kinds of the data. Please check them by README.md
# 1. continuous
# 2. ordinal
# 3. categorical

print("----------------------------------------------")

# Let the wine quality be the continuous variable
data = wineq[:, :-1] # without the last column
print(data, data.shape)

target = wineq[:, -1] # the last column
print(target, target.shape)

# Make the labels
# 1. orden every label to a string number(Example: 1-red, 2-green, 3-yellow, ...)
# 2. one-hot coding

target = wineq[:, -1].long()
print(target)

print("----------------------------------------------")

target_onehot = torch.zeros(target.shape[0], 10)
# target_onehot is 2-dim, we add one dimension using unsqueeze
print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))

target_unsqueezed = target.unsqueeze(1)
print(target_unsqueezed)
# Now the data are changed from 1-dim(4898) to 2-dim(4898x1)
data_mean = torch.mean(data, dim=0)
print(data_mean)

data_var = torch.var(data, dim=0)
print(data_var)

data_normalized = (data - data_mean) / torch.sqrt(data_var)
print(data_normalized)

print("----------------------------------------------")

# Now we want to judge which one is good, and which one is bad
bad_indexes = torch.le(target, 3) # .le can judge which one is less or equal than 3.
print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())

bad_data = data[bad_indexes]
print(bad_data.shape)

print("----------------------------------------------")

# Calculate the good, mittle and bad wine
bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# Total sulfur_threshold as the standard for good/bad wine
total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

# Get the indexes of the good wine
actual_indexes = torch.gt(target, 5)
print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_matches/n_predicted, n_matches/n_actual)