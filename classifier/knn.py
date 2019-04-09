#! -*-coding:utf8-*-
import numpy as np

class KNearestNeighbor(object):
    """kNN classifier"""
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        train_2 = np.sum(np.square(self.X_train.T), axis=0).reshape(1, -1)
        test_2 = np.sum(np.square(X), axis=1).reshape(-1, 1)
        multi = np.multiply(np.dot(X, self.X_train.T), -2)
        dists = train_2 + test_2 + multi
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sort = np.argsort(dists[i, :])
            index = sort[:k]
            closest_y = self.y_train[index]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
