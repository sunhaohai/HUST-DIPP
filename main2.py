#! -*-coding:utf8-*-
import numpy as np
from layers.linear import Linear
from loss.CrossEntropy import CrossEntropy

step = 1000


class Model:
    def __init__(self):
        self.layers = [
            Linear(20, 30),
            Linear(30, 40),
            Linear(40, 20),
            CrossEntropy()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss):
        pass

    def step(self, x):
        loss = self.forward(x)
        self.backward(loss)


def main():
    model = Model()
    for step_num in range(step):
        inputs = None
        labels = None
        model.layers[-1].labels = labels
        model.step(inputs)


if __name__ == '__main':
    main()