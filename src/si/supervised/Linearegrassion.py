import numpy as np
from .Modelo import Model
from src.si.data.dataset import Dataset
from src.si.util.util import mse


class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.num_iterations = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        if self.gd:
            self.train_gd(X, Y)
        else:
            self.train_closed(X, Y)
        self.is_fited = True

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.num_iterations):
            grad = 1/m*(X.dot(self.theta)-Y).dot(X)
            self.theta -= self.lr*grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def train_closed(self, X, Y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, x):
        assert self.is_fited
        _X = np.hstack(([1], x))
        return np.dot(self.theta, _X)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(y_pred, self.Y)


class LinearRegressionReg(LinearRegression):
    pass
