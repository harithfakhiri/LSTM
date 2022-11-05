import numpy as np
import pandas as pd
from scipy.special import expit

def preprocess(df):
    # dates = np.flip(df['Date'].to_numpy())
    opens = np.array(df['Open'].to_numpy())
    highs = np.array(df['High'].to_numpy())
    lows = np.array(df['Low'].to_numpy())
    closes = np.array(df['Close'].to_numpy())
    volumes = np.array(df['Volume'].to_numpy())

    return opens, highs, lows, closes, volumes

# directory = os.path.dirname(os.path.abspath(__file__))
# path = os.path.join(directory, "../dataset", "ETH-USD-Test.csv")
# # print(path)
# df_1 = pd.read_csv(path)
# # print(df_1)
# dates, opens, highs, lows, closes, volumes, market_caps = preprocess(df_1)

def relu(X):
    return max(0, X)

def sigmoid(X):
    return float(1/(1+np.exp(-X)))

def softmax(X):
    return lambda x: np.exp(x - np.max(X)) / np.sum(np.exp(x - np.max(X)))

def calculate(X, func_name):
    if (func_name.lower() == "relu"):
        return relu(X)
    elif (func_name.lower() == "sigmoid"):
        return sigmoid(X)
    elif (func_name.lower() == "softmax"):
        return expit(X) / np.sum(expit(X), axis=0)
    elif (func_name.lower() == "linear"):
        return X
    else:
        raise Exception("Unrecognize function name")
