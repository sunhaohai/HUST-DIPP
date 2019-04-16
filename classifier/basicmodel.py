# -*- coding:utf-8 -*-
# author:zjl
import pickle
import numpy as np

class BasicModel(object):
    def __init__(self):
        self._loss = 10000
        self.acc = 0
        self.best_acc = 0
        self.test_loss = 10000
        self.test_acc = 0

    def update(self, loss=None, acc=None):
        if loss:
            self._loss = loss
        if acc:
            self.acc = acc
            if acc > self.best_acc:
                self.best_acc = acc

    def save(self, path):
        layers = self.layers
        dictionary = {}
        dictionary['layers'] = []
        dictionary['data'] = [self._loss, self.acc, self.best_acc, self.test_loss, self.test_acc]
        package = []
        for layer in layers:
            if hasattr(layer, 'parameters'):
                package.append(layer.parameters)
            dictionary['layers'].append(str(layer))
        dictionary['parameters'] = package
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)

    def load(self, path):
        with open(path, 'rb') as f:
            dictionary = pickle.load(f)
            [self._loss, self.acc, self.best_acc, self.test_loss, self.test_acc] = dictionary['data']
        index = 0
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                layer.parameters = dictionary['parameters'][index]
                index += 1
