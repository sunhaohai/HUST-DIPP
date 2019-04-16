# -*- coding:utf-8 -*-
# author:zjl
import numpy as np

class Dropout(object):
    def __repr__(self):
        return "Dropout"

    def __init__(self, rate):
        self.rate = rate
        self.open = True
        self.grad = None
        self.z = None

    def forward(self, x):
        if self.open:
            size = x.shape[1]
            mask = np.random.binomial(1, 1-self.rate, (1, size))
            mask = np.where(mask, 1/(1-self.rate), 0)
            self.grad = mask.repeat(x.shape[0], axis=0)
            self.z = np.multiply(x, mask)
            return self.z
        else:
            return x

    def backprop(self, back_layer, optimizer=None):
        if self.open:
            self.grad = back_layer.grad * self.grad
            return self.grad
        else:
            self.grad = back_layer.grad
            return self.grad
