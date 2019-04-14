#! -*-coding:utf8-*-
"""多层线性分类器"""
import numpy as np

class MLP(object):
    """多层线性分类器"""
    def __init__(self, hidden_layer_num, hidden_size, classes_num):
        """
        hidden_layer_num: 隐藏层数目
        hidden_size: 隐藏层大小
        classes_num: 分类数目
        """
        self.hidden_layer_num = hidden_layer_num
        self.hidden_size = hidden_size
        self.classes_num = classes_num
    
    def train(self, x, y, optimizer):
        """模型训练
        x: 输入 
        y: label
        optimizer: 优化器
        """
        pass 
    
    def predict(self, x):
        """预测
        x: 输入
        """
        pass
