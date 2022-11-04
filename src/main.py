from utils import preprocess
from Layers.LSTM import LSTM
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


layer = LSTM(input_size = 32, num_cell = 2)

pred = layer.predict(opens)