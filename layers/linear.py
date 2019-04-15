#! -*-coding:utf8-*-
import numpy as np


class Linear:
    def __repr__(self):
        return "linear"

    def __init__(self, input, output):

        self.parameters = [
            np.random.randn(input, output),
            np.random.randn(1, output)
            ]

        self.forward = lambda x: np.dot(x, self.parameters[0]) + self.parameters[1]

        self.backward = None

        self.grad = None