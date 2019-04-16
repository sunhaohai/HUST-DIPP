#! -*-coding:utf8-*-
import numpy as np


class Linear(object):
    def __repr__(self):
        return "linear"

    def __init__(self, input, output):

        self.parameters = [
            np.random.randn(input, output),
            np.random.randn(1, output)
            ]

        self.z = None # 这一层的输出
        self.grad = None
        self.db = None
        self.dw = None
        self.batch_size = None
    
    def forward(self, x):
        self.batch_size = x.shape[0]
        self.z = np.dot(x, self.parameters[0]) + self.parameters[1]
        self.grad = self.parameters[0]
        self.dw = np.transpose(x)
        self.db = np.zeros(self.parameters[1].shape)
        return self.z
    
    def backprop(self, back_layer, optimizer=None):
        self.grad = np.dot(back_layer.grad, self.parameters[0].transpose())
        self.dw = np.dot(self.dw, back_layer.grad)
        self.db = np.sum(back_layer.grad, axis=0)
        if optimizer:
            self.parameters[0] = optimizer.update_parameter(self.parameters[0], self.dw, self.batch_size)
            self.parameters[1] = optimizer.update_parameter(self.parameters[1], self.db, self.batch_size)
        
        return self.grad


if __name__ == '__main__':
    x = np.array([[1, 2, 3],[4, 5, 6]])
    l = Linear(3, 10)
    print(l.forward(x).shape)
