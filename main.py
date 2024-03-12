import json 
import os 
import connectBinance 

import_path = "settings.json"

def get_settings(import_path): 
    if os.path.exists(import_path):
        file = open(import_path, "r")
        project_settings = json.load(file)
        file.close()
        return project_settings
    else: 
        return ImportError

if __name__ == "__main__": 
    project_settings = get_settings(import_path)

    print("Project Settings:", project_settings)

    api_key = project_settings["BinanceKeys"]["API_KEY"]
    secret_key = project_settings["BinanceKeys"]["Secret_Key"]


    candles=connectBinance.candlestick_data("ETHBTC","1h",50)
    print(candles)

 
