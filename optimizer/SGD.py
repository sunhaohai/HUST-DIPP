#! -*- coding:utf8 -*-

class SGD(object):
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def update_parameter(self, w, dw, batch_size):
        return w - self.lr/batch_size * dw