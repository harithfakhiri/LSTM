import sys
sys.path.append('../src')
from utils import *
import numpy as np

class LSTM:
    def __init__(self, num_cell, input_size) -> None:
        self.num_cell = num_cell
        self.input_size = input_size

        self.recurrent_weight = {}
        # 
        self.recurrent_weight["uf"] = np.random.rand(self.input_size)
        self.recurrent_weight["ui"] = np.random.rand(1, self.input_size)
        self.recurrent_weight["uc"] = np.random.rand(1, self.input_size)
        self.recurrent_weight["uo"] = np.random.rand(1, self.input_size)

        self.weight = {}
        self.weight["wf"] = np.random.rand(1, 1)
        self.weight["wi"] = np.random.rand(1, 1)
        self.weight["wc"] = np.random.rand(1, 1)
        self.weight["wo"] = np.random.rand(1, 1)

        self.bias = {}
        self.bias["f"] = np.zeros(1)
        self.bias["i"] = np.zeros(1)
        self.bias["c"] = np.zeros(1)
        self.bias["o"] = np.zeros(1)

        self.previous_c = np.zeros((1, 1), dtype=int)
        self.previous_h = np.zeros((1, 1), dtype=int)

        # self.activate_func = ActivationFunc

        self.result = {}
        self.x = [[]]
    
    def forgetGate(self, t):
        u_x = np.dot(self.recurrent_weight["uf"], self.x)
        nett = u_x + np.dot(self.weight["wf"], self.previous_h) + self.bias["f"]
        print(nett)
        self.result["f"+str(t)] = sigmoid(nett[0])
        print("forget gate", self.result["f"+str(t)])
    

    def inputGate(self, t):
        u_x = np.dot(self.recurrent_weight["ui"], self.x)
        nett = u_x + np.dot(self.weight["wi"], self.previous_h) + self.bias["i"] 
        self.result["i"+str(t)] = sigmoid(nett[0])
        print("input gate", self.result["i"+str(t)])

        u_x = np.dot(self.recurrent_weight["uc"], self.x)
        nett = u_x + np.dot(self.weight["wc"], self.previous_h) + self.bias["c"] 
        self.result["candidate"+str(t)] = np.tanh(nett[0])
        
        print("candidate", self.result["candidate"+str(t)])

    def cellState(self, t):
        f_c = np.dot(self.result["f"+str(t)], self.previous_c)
        i_candidate = np.dot(self.result["i"+str(t)], self.result["candidate"+str(t)])
        self.result["c"+str(t)] = f_c + i_candidate

        print("cell state ", self.result["c"+str(t)])

    def outputGate(self, t):        
        u_x = np.dot(self.recurrent_weight["uo"], self.x)
        nett = u_x + np.dot(self.weight["wo"], self.previous_h) + self.bias["o"] 
        self.result["o"+str(t)] = sigmoid(nett[0])

        self.result['h'+str(t)] = np.multiply(self.result["o"+str(t)], np.tanh(self.result["c"+str(t)]))

        print("output gate ", self.result["o"+str(t)])
        print("hidden state ", self.result['h'+str(t)])
    
    def forward(self, t):
        self.forgetGate(t)
        self.inputGate(t)
        self.cellState(t)
        self.outputGate(t)

        self.previous_h = self.result['h'+str(t)]
        self.previous_c = self.result['c'+str(t)]

        print(f"OUTPUT: {softmax(self.result['h'+str(t)])}")

        return self.result["h"+str(t)]

        
    def predict(self, input):

        for t in range(0, self.num_cell):
            self.x = input[t:t+self.input_size]
            curr_output = self.forward(t)
            print(f"OUTPUT {t} : {curr_output}")

        return curr_output