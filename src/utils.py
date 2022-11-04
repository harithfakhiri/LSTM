import numpy as np
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