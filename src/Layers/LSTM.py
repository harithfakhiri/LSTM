from src.utils import ActivationFunc
import numpy as np

class LSTM:
    def __init__(self, num_cell, input_size) -> None:
        self.num_cell = num_cell
        self.input_size = input_size

        self.weight = {}
        # 
        self.weight["uf"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["ui"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["uc"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["uo"] = np.random.rand(self.num_cell, self.input_size)

        self.weight["wf"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["wi"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["wc"] = np.random.rand(self.num_cell, self.input_size)
        self.weight["wo"] = np.random.rand(self.num_cell, self.input_size)

        self.bias = {}
        self.bias["f"] = np.zeros(self.num_cell, self.input_size)
        self.bias["i"] = np.zeros(self.num_cell, self.input_size)
        self.bias["c"] = np.zeros(self.num_cell, self.input_size)
        self.bias["o"] = np.zeros(self.num_cell, self.input_size)

        self.previous_c = np.zeros((self.n_cell, 1))
        self.previous_h = np.zeros((self.n_cell, 1))

        self.activate_func = ActivationFunc

        