import pandas as pd 
import numpy as np 
from binance.spot import Spot
from datetime import datetime,timedelta
import mplfinance as mpf
import os
import json
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

DAYS=5 
end = datetime.now()
end=end-timedelta(days=0)
start= end-timedelta(days=DAYS)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

klines = client.klines('BTCUSDT','1h',startTime=start_ms)
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


