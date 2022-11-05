import numpy as np
from utils import *


class Dense:
    def __init__(self, neuron, activation, input_size=None,weight=None):
        self.neuron = neuron
        self.input_size = input_size

        if (activation.lower() in ["relu", "sigmoid", "linear", "softmax"]):
            self.activation = activation
        else:
            raise Exception("Unrecognize activation function name")
        
        self.input = []
        self.weight = []
        self.output = []

    def set_input(self, input):
        self.input = input
        self.input_size = input.shape
    
    def initialize_weight(self):
        # initialize weight after input is defined
        self.weight = np.random.rand(np.array(self.input).shape[-1]).tolist()
        # self.weight = np.random.random((self.input_size + 1, self.neuron))
    
    def forward(self, input:np.array):
        if (len(input) != 1):
            raise Exception("Input needs to be 1D array")
        if (len(self.weight) == 0):
            raise Exception("Weight is not defined")

        result = []
        print("============== DENSE ================")

        print(f"INPUT  DENSE : {self.input}")
        print(f"WEIGHT DENSE : {self.weight}")

        dot_product = np.dot(np.array(self.weight).reshape((1,np.array(self.weight).shape[0])), np.array(self.input).reshape(-1,1))
        for x in dot_product:
            for y in x:
                result.append([calculate(y, self.activation)])

        self.output = result

        print(f"OUTPUT DENSE : {self.output}")
        print("--------------------------------------")

        return result
