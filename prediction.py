import pandas as pd 
import pandas_ta as ta
import numpy as np 
from binance.spot import Spot
from datetime import datetime,timedelta
import mplfinance as mpf
import os
import json
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


import_path = "settings.json"

def get_settings(import_path): 
    if os.path.exists(import_path):
        file = open(import_path, "r")
        project_settings = json.load(file)
        file.close()
        return project_settings
    else: 
        return ImportError
    
project_settings = get_settings(import_path)
api_key=project_settings["BinanceKeys"]["API_KEY"]
secret_key=project_settings["BinanceKeys"]["Secret_Key"]
#create client 
client=Spot(
        api_key=api_key,
        api_secret=secret_key,
        base_url="https://testnet.binance.vision"
    )

DAYS=1090
end = datetime.now()
end=end-timedelta(days=0)
start= end-timedelta(days=DAYS)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

klines = client.klines('BTCUSDT','1m',startTime=start_ms)
data=pd.DataFrame(
    data=[row[1:7] for row in klines ],
    columns=['open','high','low','close','volume','time'],
).set_index('time')
data.index=pd.to_datetime(data.index+1, unit='ms')
data = data.sort_index()
data=data.apply(pd.to_numeric,axis=1)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['close'], label='Close Price', color='blue')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('BTCUSDT Close Price Over Time')
plt.legend()
plt.grid(True)
plt.show()
mpf.plot(data, type='candle', style='charles', volume=True, ylabel='Price', ylabel_lower='Volume')


data[data['volume']==0]
data[data['high']== data['low']]



data["RSI"] = ta.rsi(data["close"] , length=16)
data['CCI'] = ta.cci(data['high'],data['low'],data['close'],length=16)
data['AO'] = ta.ao(data['high'],data['low'])
data['MOM'] = ta.mom(data['close'], length=16)

a = ta.macd(data['close'])
data = data.join(a)
print(data)


data['ATR'] = ta.atr(data['high'],data['low'], data['close'], length=16)
data['BOP'] = ta.bop(data['open'],data['high'], data['low'],data['close'], length=16)
data['RVI']= ta.rvi(data['close'])
a = ta.dm(data['high'],data['low'], length=16)
data = data.join(a)
a= ta.stoch(data['high'], data['low'],data['close'])
data= data.join(a)
a=ta.stochrsi(data['close'], length=16)
data=data.join(a)
data['WPR'] = ta.willr(data['high'], data['low'],data['close'], length=16)

data.isna().sum()
data.dropna(inplace=True)
data.isna().sum()
data.reset_index(drop=True, inplace=True)





#Target flexible way
pipdiff = 200*1e-4 #for TP
SLTPRatio = 2 #pipdiff/Ratio gives SL
def mytarget(barsupfront, df1):
    global pipdiff  # Using the global pipdiff variable
    length = len(df1)
    high = list(df1['high'])
    low = list(df1['low'])
    close = list(df1['close'])
    open = list(df1['open'])
    trendcat = [None] * length
    for line in range(length - barsupfront - 2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(1, barsupfront + 2):
            value1 = open[line + 1] - low[line + i]
            value2 = open[line + 1] - high[line + i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)
        if (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff / SLTPRatio)):
            trendcat[line] = 2  # uptrend
        elif (valueOpenLow <= (pipdiff / SLTPRatio)) and (-valueOpenHigh >= pipdiff):
            trendcat[line] = 1  # downtrend
        else:
            trendcat[line] = 0  # no clear trend
    return trendcat


data['Target'] = mytarget(50, data)
data['Target'].hist()
#df.tail(20)
#df['Target'] = df['Target'].astype(int)


data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data.tail()


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

attributes = ['RSI', 'CCI', 'AO', 'MOM', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATR',
       'BOP', 'RVI', 'DMP_16', 'DMN_16', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
       'STOCHRSIk_16_14_3_3', 'STOCHRSId_16_14_3_3', 'WPR']

attributes = ['MACDs_12_26_9', 'ATR', 'DMP_16']


X = data[attributes]
y = data['Target']

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost classifier model
model = XGBClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the training and testing data
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Compute accuracy scores for training and testing data
acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(y_test, pred_test)

# Print accuracy scores
print('****Train Results****')
print("Accuracy: {:.4%}".format(acc_train))
print('****Test Results****')
print("Accuracy: {:.4%}".format(acc_test))

# Print classification report for testing data
print('****Classification Report****')
print(classification_report(y_test, pred_test))

# Print confusion matrix for testing data
print('****Confusion Matrix****')
print(confusion_matrix(y_test, pred_test))
