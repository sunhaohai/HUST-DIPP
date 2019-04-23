#! -*-coding:utf8-*-
"""多层线性分类器"""
import numpy as np
from layers.linear import Linear
from layers.dropout import Dropout
from layers.batch_norm import Batchnorm
from loss.CrossEntropy import CrossEntropy
from function.activationFunc import Sigmoid, ReLU
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
            Linear(20, 32),
            Batchnorm(32),
            ReLU(),
            Dropout(0.5),
            Linear(32, 64),
            Batchnorm(64),
            ReLU(),
            Dropout(0.5),
            Linear(64, 32),
            Batchnorm(32),
            ReLU(),
            Dropout(0.5),
            Linear(32, 4),
            Sigmoid(),
            self.LossFunc
        ]
        self.parameters = {}
        for l in self.layers:
            if hasattr(l, 'parameters'):
                self.parameters[id(l)] = l.parameters

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

    def train(self, x, y, optimizer, l2_reg_lambda=0):
        """模型训练
        x: 输入 
        y: label
        optimizer: 优化器
        """
        for layer in self.layers:
            if str(layer) == "Dropout":
                layer.open = True
            if str(layer) == "Batchnorm":
                layer.train = True
        self.loss(x, y)
        self.update(self._loss, self.acc)
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[-i-1].backprop()
            self.layers[-i-1].backprop(self.layers[-i], optimizer, l2_reg_lambda)

    def predict(self, x):
        """预测
        x: 输入
        """
        for layer in self.layers:
            if str(layer) == "Dropout":
                layer.open = False
            if str(layer) == "Batchnorm":
                layer.train = False
        x = self.forward(x)
        return np.argmax(x, axis=1)
