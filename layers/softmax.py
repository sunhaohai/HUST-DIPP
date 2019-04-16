#! -*- coding:utf8 -*-

import numpy as np 

class Softmax(object):
    def __init__(self):
        self.grad = None
        self.z = None
    
    def forward(self, x):
        self.z = self._softmax(x)
        return self.z

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def gradient_softmax(grandient_last, input, output):
    """softmax层的梯度"""
    pass
