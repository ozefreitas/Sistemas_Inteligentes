import numpy as np
from src.si.util.activation import *
from src.si.util.metrics import mse, mse_prime


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, lr):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, inputsize, outputsize):
        self.weights = np.random.rand(inputsize, outputsize) - 0.5
        self.bias = np.zeros((1, outputsize))

    def setWeights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # dE/dW ? X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # dE/dX
        input_error = np.dot(output_error, self.weights.T)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation):
        self.function = activation

    def forward(self,input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def backward(self,output_error, lr):
        return np.multiply(self.function.prime(self.input),output_error)


class NN:
    def __init__(self, epochs=1000, lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            #forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")
        if not self.verbose:
            print(f"error={err}")
        self.is_fited = True

    def predict(self,input_data):
        assert self.is_fited
        output = input_data
        for layer in self.layers:
            output= layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fited, 'Model must be fitted'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)
