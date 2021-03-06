#! -*-coding:utf8-*-

import numpy as np 


class CrossEntropy(object):
    def __repr__(self):
        return "CrossEntropy"

    def __init__(self):
        self.z = None
        self._loss = None
        self.s = None
        self.grad = None

    def loss(self, x, y):
        self._loss = np.sum(-np.log(self.s[range(x.shape[0]), list(y)])) / x.shape[0]
        self.grad[range(x.shape[0]), list(y)] -= 1
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
