#! -*- coding:utf8 -*-
import numpy as np


class Batchnorm(object):
    def __repr__(self):
        return "Batchnorm"

    def __init__(self, d, momentum=0.9, eps=1e-5):
        self.train = True
        self.parameters = None
        self.eps = eps
        self.momentum = momentum
        self.dims = d
        self.grad = None
        self.x = None
        self.z = None
        self._z = None
        self.gamma = np.random.randn(1, self.dims)
        self.beta = np.random.randn(1, self.dims)
        self.global_mean = np.random.randn(1, self.dims)
        self.global_var = np.random.randn(1, self.dims)
        self.parameters = [self.gamma, self.beta]
        self.mean = None
        self.var = None
        pass

    def forward(self, x):
        # if 'dims' not in dir(Batchnorm):
        self.x = x
        if self.train:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.mean = mean
            self.var = var
            _z = (x - mean)/np.sqrt(var + self.eps)
            self._z = _z
            self.z = self.gamma * _z + self.beta
            self.global_mean = self.momentum * self.global_mean + (1 - self.momentum) * mean
            self.global_var = self.momentum * self.global_var + (1 - self.momentum) * var
        else:
            self.z = self.gamma * (x - self.global_mean)/np.sqrt(self.global_var + self.eps) + self.beta
        return self.z


    def backprop(self, back_layer, optimizer=None, l2_reg_lambda=0):
        if self.train:
            # self.grad = back_layer.grad * self.grad
            dout = back_layer.grad

            N = self.x.shape[0]
            dout_ = self.gamma * dout
            dvar = np.sum(dout_ * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5, axis=0)
            dx_ = 1 / np.sqrt(self.var + self.eps)
            dvar_ = 2 * (self.x - self.mean) / N

            # intermediate for convenient calculation
            di = dout_ * dx_ + dvar * dvar_
            dmean = -1 * np.sum(di, axis=0)
            dmean_ = np.ones_like(self.x) / N

            dx = di + dmean * dmean_
            dgamma = np.sum(dout * self._z, axis=0)
            dbeta = np.sum(dout, axis=0)

            if optimizer:
                self.parameters[0] = optimizer.update_parameter(self.parameters[0], dgamma, N, id(self), 0)
                self.parameters[1] = optimizer.update_parameter(self.parameters[1], dbeta, N, id(self), 1)
            self.grad = dx
            return self.grad
        else:
            self.grad = back_layer.grad
            return self.grad