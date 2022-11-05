from utils import preprocess
from Layers.LSTM import LSTM
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from sequential import Sequential
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # only for normalize the data

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "../dataset", "ETH-USD-Train.csv")
print(path)
df_1 = pd.read_csv(path)
# print(df_1)
opens, highs, lows, closes, volumes = preprocess(df_1)
# print(opens)
print(opens)

scaler = MinMaxScaler(feature_range = (0, 1)) # scale the data

open_scaled = scaler.fit_transform(np.array(opens).reshape(-1,1))


lstm = LSTM(input_size = 32, num_cell = 5)
flatten = Flatten()
dense = Dense(neuron=10, activation="sigmoid")

sequential = Sequential(LSTM=lstm, Flatten=flatten, Dense=[dense])
predicted_open = sequential.predict(open_scaled, "Opens")

rescaled_open = scaler.inverse_transform(predicted_open)
print("rescaled open", rescaled_open)