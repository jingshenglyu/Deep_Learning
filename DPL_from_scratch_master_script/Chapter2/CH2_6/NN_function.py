#!/usr/bin/env python
# coding=utf-8
# 
# @Author       : Jingsheng Lyu
# @Date         : 2020-07-01 23:17:27
# @LastEditors  : Jingsheng Lyu
# @LastEditTime : 2020-07-01 23:18:09
# @FilePath     : /Deep_Learning/Chapter2/CH2_6/NN_function.py
# @Github       : https://github.com/jingshenglyu
# @Web          : https://jingshenglyu.github.io/
# @E-Mail       : jingshenglyu@gmail.com
# 

import sys, os
import numpy as np
import pickle
sys.path.append(os.pardir) # Why we do this? We want to import the parent's folder
from dataset.mnist import load_mnist # import dataset

# function definition
def softmax(x):
    c = np.max(x)
    return np.exp(x-c) / np.sum(np.exp(x-c))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:  
        # the weights from learning will be saved in sample_weight.pkl
        # In this .pkl file, the weights and the bias are saved by dict{}.
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y