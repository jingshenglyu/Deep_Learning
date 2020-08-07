#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 08.07.20 19:11
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH0_2_Derivative.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import torch

points = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
# A torch.Storage is a contiguous, one-dimensional array of a single data type.
print(points.storage())

print("-----------------------------")

# Index of storage is always 1-dim
print("Points' storage is", float((points.storage()[1])))
# print(points.storage()[0][1]) !!!wrong

print("-----------------------------")

# storage_offset
second_point = points[1]

print(second_point)
print("storage offset is", second_point.storage_offset())

second_storage = points[0][1]
print("Second storage of points is", second_storage)

print("-----------------------------")

# size and shape are same
print("size of second_point is", second_point.size())
print("shape of second_point is", second_point.shape)

# Stride is a tuple
print("stride of points is", points.stride())
print("stride of second_points is", second_point.stride())

print("-----------------------------")

# points is the father's tensor and second_point is the child' tensor
second_point[0] = 10.0
print(second_point)
print(points)
print("\n *It means second_point and points are changed at same time.* \n")

# If we want to avoid this same changed. We can use .clone()
second_point_clone = points[1].clone()
second_point_clone[0] = 15.0
print(second_point_clone)
print(points)

print("-----------------------------")

# Transposition
points_t = points.t()
print(points_t)

# Verify the id for the original and the transposition
print(id(points) == id(points_t))
print(id(points.storage()) == id(points_t.storage()))
print(points.storage())
print(points_t.storage())

# Transposition for n-dim tensor
print("3-dim tensor")
three_dim_tensor = torch.randn(3,4,5)
print(three_dim_tensor)
print(three_dim_tensor.shape, three_dim_tensor.stride())

print("\n")

print("3-dim tensor for transposition")
# only transposition 0th and 2nd dim
three_dim_tensor_t = three_dim_tensor.transpose(0, 2)
print(three_dim_tensor_t)
print(three_dim_tensor_t.shape, three_dim_tensor_t.stride())

print("-----------------------------")

# confirm the continuity
print("Continuity can improve data localization and performae.\n"
      "Contigous is better")
print(points.is_contiguous(), points_t.is_contiguous())

# Cange the continuity
points_t_cont = points_t.contiguous()
print(points_t_cont.is_contiguous())




