#! -*- coding:utf8 -*-

import numpy as np 

def batch_norm(x, run_mean, run_var, momentum, gamma, beta, train=True):
    """batch_norm层
    input:
      x: 输入
      run_mean: 前面数据的平均值
      run_var: 前面数据的方差
      momentum: 衰减速率
      gamma: 参数
      beta: 参数
    output:
      batch norm 后的数据
    """
    pass

def gradient_batch_norm(gradient_last, input, output):
    """batch_norm 的梯度
    output:
      gg: gamma梯度
      beta: beta梯度
    """
    pass