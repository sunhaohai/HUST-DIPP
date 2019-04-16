#! -*-coding:utf8-*-
"""多层线性分类器"""
import numpy as np
from layers.linear import Linear
from loss.CrossEntropy import CrossEntropy
from function.activationFunc import Sigmoid
from optimizer.SGD import SGD

class MLP(object):
    """多层线性分类器"""
    def __init__(self, layer_size):
        """
        hidden_layer_num: 隐藏层数目
        hidden_size: 隐藏层大小
        classes_num: 分类数目
        """
        self.LossFunc = CrossEntropy()
        self.layers = []
        for l in layer_size:
            self.layers.append(Linear(l[0], l[1]))
            self.layers.append(Sigmoid())
        self.layers.append(self.LossFunc)
        self._loss = None
        self.acc = None

    def forward(self, x):
        tmp = x
        for layer in self.layers:
            tmp = layer.forward(tmp)
        return tmp
    
    def loss(self, x, y):
        x = self.forward(x)
        self.acc = np.mean(y==np.argmax(x, axis=1))
        self._loss = self.LossFunc.loss(x, y)
        return self._loss

    def train(self, x, y, optimizer):
        """模型训练
        x: 输入 
        y: label
        optimizer: 优化器
        """
        self.loss(x, y)
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[-i-1].backprop()
            self.layers[-i-1].backprop(self.layers[-i], optimizer)
            #print(self.layers[-i-1].grad)
        
    
    def predict(self, x):
        """预测
        x: 输入
        """
        x = self.forward(x)
        return np.argmax(x, axis=1)
