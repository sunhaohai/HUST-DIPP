#! -*- coding:utf8 -*-
import numpy as np


class SGD(object):
    def __init__(self, parameters, lr=0.001, momentum=0, gamma=0.99999, method='fixed', min_lr=1e-3, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.gamma = gamma
        self.method = method
        self.step = 0
        self.parameters = parameters
        self.m = {}
        self.min_lr = min_lr
        self.nesterov = nesterov
        for key, v in self.parameters.items():
            if key not in self.m.keys():
                self.m[key] = []
            for param in v:
                self.m[key].append(np.zeros(param.shape))
    
    def update_parameter(self, w, dw, batch_size, obj_id, index):
        _lr = self.lr
        if self.method == 'exp':
            _lr = self.lr * (self.gamma ** self.step)

        if _lr < self.min_lr:
            _lr = self.min_lr
        
        self.m[obj_id][index] = self.momentum * self.m[obj_id][index] + dw / batch_size 
        self.step += 1
        return w - _lr * self.m[obj_id][index]
