import numpy as np

class ActivationFunc:
    def relu(self, X):
        return max(0, X)

    def sigmoid(self, X):
        return float(1/(1+np.exp(-X)))

    def softmax(self, X):
        return np.exp(X)/np.exp(X).sum(axis=1, keepdims=True)

    def calculate(self, X, func_name):
        if (func_name.lower() == "relu"):
            return self.relu(X)
        elif (func_name.lower() == "sigmoid"):
            return self.sigmoid(X)
        elif (func_name.lower() == "softmax"):
            return self.softmax(X)
        else:
            raise Exception("Unrecognize function name")

