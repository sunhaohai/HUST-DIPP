#! -*- coding:utf8 -*-

import numpy as np 

class Sigmoid(object):
    def __init__(self):
        self.grad = None
        self.z = None
    
    def forward(self, x):
        self.z = self._sigmoid(x)
        self.grad = self.z * (1 - self.z)
        return self.z
    
    def backprop(self, back_layer, optimizer=None):
        self.grad = back_layer.grad * self.grad
        return self.grad
    
    def _sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

def relu(x):
    """relu 激活函数"""
    return np.where(x<0, 0, x)

def gradient_relu(x):
    """relu梯度"""
    return np.where(x<0, 0, 1)
