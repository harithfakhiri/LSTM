import sys
sys.path.append('../src')
from utils import *
import numpy as np

class LSTM:
    def __init__(self, num_cell, input_size, return_sequence=False) -> None:
        self.num_cell = num_cell
        self.input_size = input_size

        self.recurrent_weight = {}
        # 
        self.recurrent_weight["uf"] = np.random.uniform(-1, 1, (self.num_cell, self.input_size))
        self.recurrent_weight["ui"] = np.random.uniform(-1, 1, (self.num_cell, self.input_size))
        self.recurrent_weight["uc"] = np.random.uniform(-1, 1, (self.num_cell, self.input_size))
        self.recurrent_weight["uo"] = np.random.uniform(-1, 1, (self.num_cell, self.input_size))

        self.weight = {}
        self.weight["wf"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.weight["wi"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.weight["wc"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.weight["wo"] =  np.random.uniform(-1, 1, (self.num_cell, 1))

        self.bias = {}
        self.bias["f"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.bias["i"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.bias["c"] = np.random.uniform(-1, 1, (self.num_cell, 1))
        self.bias["o"] = np.random.uniform(-1, 1, (self.num_cell, 1))

        # print(f"BIAS: {self.bias['f']}, {self.bias['i']}, {self.bias['c']}, {self.bias['o']}")

        self.previous_c = np.zeros((self.num_cell, 1), dtype=int)
        self.previous_h = np.zeros((self.num_cell, 1), dtype=int)

        # self.activate_func = ActivationFunc

        self.result = {}
        self.input = []
        self.x = [[]]
        self.output = []
        self.is_sequence = return_sequence
    
    def forgetGate(self, t):
        self.result["f"+str(t)] = []
        for i in range(self.num_cell):
            u_x = np.dot(self.recurrent_weight["uf"][i], self.x)
            nett = u_x + np.dot(self.weight["wf"][i], self.previous_h[i]) + self.bias["f"][i]
            # print(nett)
            self.result["f"+str(t)].append(sigmoid(nett[0]))
        # print("forget gate", self.result["f"+str(t)])
    

    def inputGate(self, t):
        self.result["candidate"+str(t)] = []
        self.result["i"+str(t)] = []
        for i in range(self.num_cell):
            u_x = np.dot(self.recurrent_weight["ui"][i], self.x)
            nett = u_x + np.dot(self.weight["wi"][i], self.previous_h[i]) + self.bias["i"][i]
            self.result["i"+str(t)].append(sigmoid(nett[0]))
        # print("input gate", self.result["i"+str(t)])

            u_x = np.dot(self.recurrent_weight["uc"][i], self.x)
            nett = u_x + np.dot(self.weight["wc"][i], self.previous_h[i]) + self.bias["c"][i]
            self.result["candidate"+str(t)].append(np.tanh(nett[0]))
        
        # print("candidate", self.result["candidate"+str(t)])

    def cellState(self, t):
        self.result["c"+str(t)] = []
        for i in range(self.num_cell):
            f_c = np.dot(self.result["f"+str(t)][i], self.previous_c[i])
            i_candidate = np.dot(self.result["i"+str(t)][i], self.result["candidate"+str(t)][i])
            self.result["c"+str(t)].append(f_c + i_candidate)

        # print("cell state ", self.result["c"+str(t)])

    def outputGate(self, t):        
        self.result['h'+str(t)] = []
        self.result["o"+str(t)] = []
        for i in range(self.num_cell):
            u_x = np.dot(self.recurrent_weight["uo"][i], self.x)
            nett = u_x + np.dot(self.weight["wo"][i], self.previous_h[i]) + self.bias["o"][i]
            self.result["o"+str(t)].append(sigmoid(nett[0]))

            self.result['h'+str(t)].append(np.multiply(self.result["o"+str(t)][i], np.tanh(self.result["c"+str(t)][i])))

        # print("output gate ", self.result["o"+str(t)])
        # print("hidden state ", self.result['h'+str(t)])
    
    def forward(self, t):
        self.forgetGate(t)
        self.inputGate(t)
        self.cellState(t)
        self.outputGate(t)

        self.previous_h = self.result['h'+str(t)]
        self.previous_c = self.result['c'+str(t)]

        return self.result["h"+str(t)]

        
    def train(self, input):
        self.input = input
        print("================================ LSTM ================================")
        # for t in range(0, len(input) - self.input_size):
        for t in range(0, 3):
            print(F"TIMESTEP {t+1}")
            self.x = input[t: t+self.input_size]

            print(F"INPUT {t+1}       : \n{self.x}\n")

            curr_output = self.forward(t)
            self.output.append(np.asarray(curr_output))

            print(f"FORGET GATE  {t+1} : {self.result['f'+str(t)]}")
            print(f"INPUT GATE   {t+1} : {self.result['i'+str(t)]}")
            print(f"CANDIDATE    {t+1} : {self.result['candidate'+str(t)]}")
            print(f"CELL GATE    {t+1} : {self.result['c'+str(t)]}")
            print(f"OUTPUT GATE  {t+1} : {self.result['o'+str(t)]}")
            print(f"HIDDEN STATE {t+1} : {curr_output}")
            print("-----------------------------------------------------------------------")
        
        if (self.is_sequence): 
            return np.array(self.output)
        else:
            return np.array(curr_output)
    
    def predict(self, output, t, column):
        self.input = column
        is_sequence = self.is_sequence
        self.is_sequence = False
        if (t == 0):
            self.x = self.input[-self.input_size:]
    
        if(len(output) > 0) :
            self.x = self.x[1:len(self.x)]
            self.x.append(output[t-1])
            curr_output = self.forward(t)
            output.append(curr_output)
        else:
            curr_output = self.forward(t)
            output.append(curr_output)

        self.is_sequence = is_sequence
        return output
    
    def type_(self, id):
        #id: id layer >= 0
        return 'lstm'+str(id)+' (LSTM)'

    def getParam(self, k):
        m = len(self.x.shape)
        n = self.num_cell
        return (m+n+1)*4*n+(n+1)*k