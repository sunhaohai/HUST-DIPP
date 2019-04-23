#! -*-coding:utf8-*-

import numpy as np 


class MSELoss(object):
    def __repr__(self):
        return "MSELoss"

    def __init__(self):
        self.z = None
        self._loss = None
        self.s = None
        self.grad = None

    def loss(self, x, y):
        new_y = np.zeros(x.shape)
        new_y[range(x.shape[0]), list(y)] = 1
        self._loss = np.sum(np.square(x - new_y) / 2.0) / x.shape[0]
        self.grad = x - new_y
        return self._loss

    def forward(self, x):
        x = x - np.max(x)
        self.s = self._softmax(x)
        self.grad = self._softmax(x)
        return self.s

    def backprop(self, back_layer=None, optimizer=None, l2_reg_lambda=0):
        return self.grad

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
