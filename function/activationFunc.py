#! -*- coding:utf8 -*-

import numpy as np 


class Sigmoid(object):
    def __repr__(self):
        return "Sigmoid"

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

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


class ReLU(object):
    def __repr__(self):
        return "ReLU"

    @staticmethod
    def _relu(x):
        return np.where(x < 0, 0, x)

    def __init__(self):
        self.grad = None
        self.z = None

    def forward(self, x):
        self.z = self._relu(x)
        self.grad = np.where(x < 0, 0, 1)
        return self.z

    def backprop(self, back_layer, optimizer=None):
        self.grad = back_layer.grad * self.grad
        return self.grad


if __name__ == "__main__":
    a =ReLU()
    print(a)