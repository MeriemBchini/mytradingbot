from binance.spot import Spot ;
import pandas as pd
import mplfinance as mpf


#get the status function 
def binance_status(): 
    status=Spot().account_status()
    if status["status"]==0 :
        return True 
    else: 
        raise ConnectionError 
    
#establish connection an get account 
def binance_account(api_key, secret_key):
    return Spot(
        api_key=api_key,
        api_secret=secret_key,
        base_url="https://testnet.binance.vision"
    ).account()


#check if testnet work
def test_testnet():
    client= Spot(base_url="https://testnet.binance.vision")
    print(client.time())

#function to get candelstick data 
def candlestick_data(symbol,timeframe,qty): 
    #get raw data 
    raw_data=Spot().klines(symbol=symbol,interval=timeframe,limit=qty) 
    converted_data=[]
    for candlestick in raw_data: 
        converted_candlestick={
            "open_time":candlestick[0],
            "open_price":float(candlestick[1]),
            "high_price":float(candlestick[2]),
            "low_price":float(candlestick[3]), 
            "close_price":float(candlestick[4]),
            "volume": float(candlestick[5]), 
            "Close_time":(candlestick[6]),
            "Quote_asset_volume":float(candlestick[7]),
            "Number_of_trades": int(candlestick[8]),
            "Taker_buy_base_asset_volume":float(candlestick[9]), 
            "Taker buy quote asset volume":float(candlestick[10])
        }
        converted_data.append(converted_candlestick)
        # Create a DataFrame from the list of dictionaries
    converted_data = pd.DataFrame(converted_data)

    # Convert timestamps to datetime format
    converted_data['open_time'] = pd.to_datetime(converted_data['open_time'], unit='ms')
    converted_data['Close_time'] = pd.to_datetime(converted_data['Close_time'], unit='ms')
    converted_data = converted_data.rename(columns={
        "open_time": "Date",
        "open_price": "Open",
        "high_price": "High",
        "low_price": "Low",
        "close_price": "Close",
        "volume": "Volume"
    })

    # Visualize using mplfinance
    mpf.plot(converted_data.set_index('Close_time').tail(100), type='candle', volume=True , mav=(3,6,9),style='charles')

 
    return converted_data
    



#get symbol from base asset 
def get_asset_symbol(asset_symbol):
    symbol_dictionary=Spot().exchange_info()
    #convert this into dataframe(tabular form)
    symbol_dataframe = pd.DataFrame(symbol_dictionary["symbols"])
    #filtre all the symbols with the base asset pair
    symbol_dataframe=symbol_dataframe.loc[
        symbol_dataframe["quoteAsset"] == asset_symbol 
    ] 
    #filtre symbols that are available for trading 
    symbol_dataframe=symbol_dataframe.loc[
        symbol_dataframe["status"]== "TRADING"
    ]
    return symbol_dataframe

     

