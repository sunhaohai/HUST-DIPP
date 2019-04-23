#! -*-coding:utf8-*-

import numpy as np 


class FocalLoss(object):
    def __repr__(self):
        return "FocalLoss"

    def __init__(self, alpha=[0.76, 0.71, 0.70, 0.81], gamma=2):
        self.z = None
        self._loss = None
        self.s = None
        self.grad = None
        self.p = None
        self.alpha = alpha
        self.gamma = gamma

    def loss(self, x, y):
        self._loss = np.sum(-np.log(self.s[range(x.shape[0]), list(y)])) / x.shape[0]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if j == y[i]:
                    self.grad[i][j] = self.alpha[j] * (-self.gamma*((1-self.p[i][y[i]])**(self.gamma-1))*np.log(
                        self.p[i][y[i]])*self.p[i][y[i]]+(1-self.p[i][y[i]])**self.gamma) * (self.p[i][j]-1)
                else:
                    self.grad[i][j] = self.alpha[j] * (-self.gamma*((1-self.p[i][y[i]])**(self.gamma-1))*np.log(
                        self.p[i][y[i]])*self.p[i][y[i]]+(1-self.p[i][y[i]])**self.gamma) * self.p[i][j]
        return self._loss

    def forward(self, x):
        x = x - np.max(x)
        self.s = self._softmax(x)
        self.grad = self._softmax(x)
        self.p = self._softmax(x)
        return self.s

    def backprop(self, back_layer=None, optimizer=None, l2_reg_lambda=0):
        return self.grad

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
