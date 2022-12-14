import numpy as np


class Flatten:
    def __init__(self):
        self.shape_before = None
        self.type_ = 'flatten (Flatten)'

    def forward(self, inputs: np.array):
        print("============== FLATTEN ==============")
        print(F"INPUT FLATTEN  : {inputs}", end=" ")
        print("")
        self.shape_before = inputs.shape
        flatten_output = inputs.flatten()

        print(f"OUTPUT FLATTEN : {flatten_output}")
        print("-------------------------------------")

        return flatten_output