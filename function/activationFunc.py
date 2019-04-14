#! -*- coding:utf8 -*-

import numpy as np 

def sigmoid(x):
    """sigmoid 激活函数"""
    return 1.0/(1.0 + np.exp(-x))

def gradient_sigmoid(x):
    """sigmoid梯度"""
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """relu 激活函数"""
    return np.where(x<0, 0, x)

def gradient_relu(x):
    """relu梯度"""
    return np.where(x<0, 0, 1)