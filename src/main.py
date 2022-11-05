from utils import preprocess
from Layers.LSTM import LSTM
from Layers.Dense import Dense
from Layers.Flatten import Flatten
import os
import pandas as pd
import numpy as np

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "../dataset", "ETH-USD-Train.csv")
print(path)
df_1 = pd.read_csv(path)
# print(df_1)
opens, highs, lows, closes, volumes = preprocess(df_1)
# print(opens)


layer1 = LSTM(input_size = 32, num_cell = 5)
layer2 = Flatten()
layer3 = Dense(neuron=10, activation="sigmoid")

pred1 = layer1.predict(opens)
pred1_flatten = layer2.forward(pred1)
layer3.set_input(pred1_flatten)
layer3.initialize_weight()
output = layer3.forward(pred1_flatten)

print(f"PREDICTED OPENS: {output}")
