import datetime
import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import scipy.io as sciio
if not  os.path.exists(rf"my_data"):
    os.makedirs(rf"my_data")
ticker = pd.read_csv(r"/var/TSDB/stk/tickers.csv",
                     index_col=0).dropna().values
ticker = ["SH" + ii if ii[0] == "6" else "SZ" + ii for ii in ticker.ravel()]

nameDict = {"avgprice":"vwap","vol":"volume","adjfactor":"factor"}
base_path = r"/var/TSDB/stk/d01/e"
ts = np.fromfile("/".join([base_path,"ts"]), offset=16, dtype=np.int64) / (10 ** 9)
ts = np.array([int(datetime.datetime.fromtimestamp(i).date().strftime("%Y%m%d")) for i in ts])
dataSet = xr.Dataset()
for factor_name in os.listdir(base_path):
    if factor_name in ['adjfactor','amount','avgprice','close','high','low','open','preclose','vol']:
    # if (factor_name != "ts") and (factor_name!="universe"):
        tmp = np.fromfile("//".join([base_path,factor_name]), offset=16,
                    dtype=np.float32).reshape(-1, 8000)
        tmp = tmp[:, :len(ticker)]
        tmp = xr.DataArray(tmp,
                            dims=('date', 'symbol'),
                            coords={'date': ts[:tmp.shape[0]],"symbol":ticker})
        if factor_name not in nameDict.keys():
            dataSet[factor_name] = tmp
        else:
            dataSet[nameDict[factor_name]] = tmp
base_path = r"/var/TSDB/stk/d01/b"
ts_b = np.fromfile("/".join([base_path, "ts"]), offset=16, dtype=np.int64) / (10 ** 9)
ts_b = np.array([int(datetime.datetime.fromtimestamp(i).date().strftime("%Y%m%d")) for i in ts_b])

for factor_name in ['float_a','free_float']:
    tmp = np.fromfile("//".join([base_path,'share',factor_name]), offset=16,
                dtype=np.float32).reshape(-1, 8000)
    tmp = tmp[:, :len(ticker)]
    tmp = xr.DataArray(tmp,
                        dims=('date', 'symbol'),
                        coords={'date': ts_b[:tmp.shape[0]],"symbol":ticker})
    if factor_name not in nameDict.keys():
        dataSet[factor_name] = tmp
    else:
        dataSet[nameDict[factor_name]] = tmp

# for factor_name in ['open','high','low','close','vwap']:
#     dataSet[factor_name] = dataSet[factor_name]*dataSet['factor']
for factor_name in list(dataSet.data_vars):
    dataSet[factor_name].to_netcdf(f"my_data/{factor_name}.nc",engine='h5netcdf')