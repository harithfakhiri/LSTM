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
        self.input = [input]
        self.input_size = input.shape
        print(self.input_size)
    
    def initialize_weight(self):
        # initialize weight after input is defined
        print(f"SHAPE INPUT: {len(self.input)}")
        self.weight = np.random.rand(len(self.input)).tolist()
        # self.weight = np.random.random((self.input_size + 1, self.neuron))
    
    def forward(self, input:np.array):
        # if (len(input) != 1):
        #     raise Exception("Input needs to be 1D array")
        if (len(self.weight) == 0):
            raise Exception("Weight is not defined")

        result = []
        print("============== DENSE ================")

        print(f"INPUT  DENSE : {self.input}")
        print(f"WEIGHT DENSE : {self.weight}")

        for i in range(len(self.input[0])):
            print(f"RESHAPED WEIGHT: {np.array(self.weight).reshape((1,np.array(self.weight).shape[0]))}")
            print(f"RESHAPED INPUT: {np.array(self.input[0][i]).reshape(-1,1)}")
            dot_product = np.dot(np.array(self.weight).reshape((1,np.array(self.weight).shape[0])), np.array(self.input[0][i]).reshape(-1,1))

            result.append([calculate(dot_product, self.activation)])

        self.output = result

        print(f"OUTPUT DENSE : {self.output}")
        print("--------------------------------------")

        return result
