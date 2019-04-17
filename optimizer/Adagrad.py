#! -*- coding:utf8 -*-

import numpy as np

class Adagrad(object):
    def __init__(self, parameters, lr=0.001, gamma=0.9999, method='fixed', min_lr=1e-10):
        self.lr = lr
        self.gamma = gamma
        self.method = method
        self.step = 0
        self.parameters = parameters
        self.min_lr = min_lr
        self.g = {}
        for key, v in self.parameters.items():
            if key not in self.g.keys():
                self.g[key] = []
            for param in v:
                self.g[key].append(np.zeros(param.shape))
    
    def update_parameter(self, w, dw, batch_size, obj_id, index):
        _lr = self.lr
        if self.method == 'exp':
            _lr = self.lr * (self.gamma ** self.step)

        if _lr < self.min_lr:
            _lr = self.min_lr
        
        self.g[obj_id][index] += np.square(dw)
        self.step += 1
        return w - _lr / np.sqrt(self.g[obj_id][index] + 1e-10) * dw

        
