import numpy as np
<<<<<<< HEAD
import pandas as pd
import os

def preprocess(df):
    dates = np.flip(df['Date'].to_numpy())
    opens = np.flip(df['Open'].to_numpy())
    highs = np.flip(df['High'].to_numpy())
    lows = np.flip(df['Low'].to_numpy())
    closes = np.flip(df['Close'].to_numpy())
    volumes = [int(val.replace(',','').replace('-','0')) for val in np.flip(df['Volume'].to_numpy())]
    market_caps = [int(val.replace(',','').replace('-','0')) for val in np.flip(df['Market Cap'].to_numpy())]

    return dates, opens, highs, lows, closes, volumes, market_caps

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "../dataset", "ETH-USD-Test.csv")
print(path)
df_1 = pd.read_csv(path)
print(df_1)
dates, opens, highs, lows, closes, volumes, market_caps = preprocess(df_1)
=======

class ActivationFunc:
    def relu(self, X):
        return max(0, X)

    def sigmoid(self, X):
        return float(1/(1+np.exp(-X)))

    def softmax(self, X):
        return np.exp(X)/np.exp(X).sum(axis=1, keepdims=True)

    def calculate(self, X, func_name):
        if (func_name.lower() == "relu"):
            return self.relu(X)
        elif (func_name.lower() == "sigmoid"):
            return self.sigmoid(X)
        elif (func_name.lower() == "softmax"):
            return self.softmax(X)
        else:
            raise Exception("Unrecognize function name")

>>>>>>> 1f96f8c51ad00208361250f7efb08220af65f295
