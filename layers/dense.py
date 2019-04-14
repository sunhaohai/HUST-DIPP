#! -*-coding:utf8-*-
import numpy as np

def dense(W, X, b):
    """全连接层"""
    return np.dot(W, X) + b

def gradient_dense(gradient_last, input, output):
    """全连接层的梯度"""
    pass
