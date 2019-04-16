#! -*-coding:utf8-*-
"""多层线性分类器"""
import numpy as np
from layers.linear import Linear
from layers.dropout import Dropout
from loss.CrossEntropy import CrossEntropy
from function.activationFunc import Sigmoid, ReLU
from optimizer.SGD import SGD
from classifier.basicmodel import BasicModel


class MLP(BasicModel):
    """多层线性分类器"""
    def __init__(self):
        """
        hidden_layer_num: 隐藏层数目
        hidden_size: 隐藏层大小
        classes_num: 分类数目
        """
        super(MLP, self).__init__()
        self.LossFunc = CrossEntropy()
        self.layers = [
            Linear(20, 128),
            ReLU(),
            Dropout(0.2),
            Linear(128, 256),
            ReLU(),
            Dropout(0.5),
            Linear(256, 128),
            Sigmoid(),
            Dropout(0.2),
            Linear(128, 4),
            self.LossFunc
        ]

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
        self.update(self._loss, self.acc)
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[-i-1].backprop()
            self.layers[-i-1].backprop(self.layers[-i], optimizer)

    def predict(self, x):
        """预测
        x: 输入
        """
        x = self.forward(x)
        return np.argmax(x, axis=1)
