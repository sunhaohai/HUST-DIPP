#! -*- coding:utf8 -*-

import numpy as np 

def softmax(x, dim=None):
    """softmax 函数 dim表示做softmax的维度, 默认最后一个维度"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def gradient_softmax(grandient_last, input, output):
    """softmax层的梯度"""
    pass
