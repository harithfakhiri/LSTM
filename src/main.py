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

def main(fitur_arr, fiturname):
    scaler = MinMaxScaler(feature_range = (0, 1)) # scale the data

    input_fitur_scaled = scaler.fit_transform(np.array(fitur_arr).reshape(-1,1))

    # neuron == num_cell
    lstm = LSTM(input_size = 32, num_cell = 3, return_sequence=False)
    flatten = Flatten()
    dense = Dense(neuron = 5, activation="softmax")
    dense2 = Dense(neuron=1, activation="sigmoid")

    sequential = Sequential(LSTM=[lstm], Flatten=flatten, Dense=[dense, dense2])
    predicted_from_training = np.array([sequential.train(input_fitur_scaled, fiturname)]).reshape(1,-1)

    rescaledrescaled_prediction_train = scaler.inverse_transform(predicted_from_training)

    predictNext30 = np.array(sequential.predict(input_fitur_scaled, fiturname, 51)).reshape(1,-1)
    final_30_prediction = scaler.inverse_transform(predictNext30)
    # print("rescaled open", rescaledrescaled_prediction_train)
    # print("this is for the next 30 days", predictNext30)
    print(f"predict {fiturname} for the next 51 days in test csv")
    for i in range(len(final_30_prediction[0])):
        print("day", i+1, ":", final_30_prediction[0][i])

main(closes, "close")