#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time         : 10.07.20 19:38
# @Author       : Jingsheng Lyu
# @Site         : 
# @File         : CH0_5_Serializing.py
# @Software     : PyCharm
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com

import torch

# save points to /common/ourpoints.t
points = torch.randn(2, 3)
print(points)
torch.save(points, 'ourpoints.t')
#with open('/common/ourpoints.pt', 'wb') as f:
#    torch.save(points, f)

# load points from /common/ourpoints.t
points_load = torch.load('ourpoints.t')
print(points_load)
points_load[1, 0] = 5.0
print(points_load)