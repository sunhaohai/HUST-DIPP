#! -*-coding:utf8-*-

import numpy as np 


class CrossEntropy:
    def __repr__(self):
        return "CrossEntropy"

    def __init__(self):

        self.labels = None

        self.forward = lambda x: -np.sum(self.labels * np.log(x), axis=1, keepdims=True)

        self.backward = None

        self.grad = None