import pandas as pd 

import numpy as np 
from pandas.tseries.offsets import MonthBegin
from binance.client import Client

import json
client=Client()
info = client.get_exchange_info()
min_volume = 100
symbols=[x['symbol'] for x in info['symbols']if('ETHBTC' in x['symbol'] or 'BTCUSDT' in x['symbol'] ) ]

def getmonthlydata(symbol): 
    frame = pd.DataFrame(client.get_historical_klines(symbol,'1M','2021-01-01','2023-12-31'))

    if len(frame)>0: 
        frame=frame[[0,4]]
        frame.columns = ['Time',symbol]
        frame = frame.set_index('Time')
        frame.index = pd.to_datetime(frame.index, unit='ms')
        frame = frame.astype(float)
        return frame


data=[]
for coin in symbols:
    data.append(getmonthlydata(coin))



#concatinating data
mergeddata=pd.concat(data,axis=1)


ret_data= mergeddata.pct_change()
logret_data= np.log(ret_data+1)

def top_ret(window, top_n, date):
    ret_wd=logret_data.rolling(window).sum()
    ret_s=ret_wd.loc[date]
    top_performers=ret_s.nlargest(top_n)
    perf_ret=ret_data.loc[top_performers.name+ MonthBegin(1), top_performers.index].mean()
    return perf_ret

matrix=[]
profits=[]
windows=range(1,13)
for window in windows:
    for date in ret_data.index[1:-1]:
        profits.append(top_ret(window,5,date))
    matrix.append(profits)
    profits=[]

all_month=pd.DataFrame(matrix,index=windows)
(all_month+1).prod(axis=1)-1

