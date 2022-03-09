# from alphanet import AlphaNetV3, load_model
# from alphanet.metrics import UpDownAccuracy
import torch
import xarray as xr
from tqdm import tqdm

import os
from model import *
from rawData import TrainValTestData, TimeSeriesData

# read data
if not os.path.exists('model'):
    os.makedirs('model')
tmp = xr.open_mfdataset('my_data/*.nc', chunks={"date": 240}, parallel=True, engine='h5netcdf')#.to_array(name='feature')
tmp['vwap_adj'] = tmp['vwap']*tmp['factor']
tmp['turn'] = tmp['volume']/tmp['float_a']
tmp['free_turn'] = tmp['volume']/tmp['free_float']
tmp['close_freeturn'] = tmp['close']/tmp['free_turn']
tmp['open_turn'] = tmp['open']/tmp['turn']
tmp['volume_low'] = tmp['volume']/tmp['low']
tmp['vwap_high'] = tmp['vwap']/tmp['high']
tmp['low_high'] = tmp['low']/tmp['high']
tmp['vwap_close'] = tmp['vwap']/tmp['close']
tmp['return_1'] = (tmp['vwap_adj']/tmp['vwap_adj'].shift({'date':1})-1)
tmp['future_10_cum_return'] = (tmp['vwap_adj'].shift({'date':-11})/tmp['vwap_adj'].shift({'date':-1})-1)


tmp = tmp.to_array(name='feature')
df = tmp.stack(sample=('date', 'symbol')).to_pandas().T
df = df.dropna(subset=['return_1'])
df = df.dropna(how='all')
# df = pd.read_csv("some_data.csv")
del tmp
# compute label (future return)
df_future_return = df['vwap'].groupby('symbol').apply(lambda x : x.pct_change(-10).shift(-1)).to_frame('future_10_cum_return').reset_index()
df = df[['future_10_cum_return','open','high', 'low','close',  'vwap', 'volume', 'return_1', 'turn', 'free_turn',
       'close_freeturn', 'open_turn', 'volume_low', 'vwap_high', 'low_high',
       'vwap_close']].reset_index()
feature_num = len(df.columns)-3
# create an empty list
stock_data_list = []

# put each stock into the list using TimeSeriesData() class
security_codes = df["symbol"].unique()
from joblib import Parallel,delayed
def fillData(table_part):
    return TimeSeriesData(dates=table_part["date"].values,  # date column
                   data=table_part.iloc[:, 3:].values,  # data columns
                   labels=table_part["future_10_cum_return"].values)

stock_data_list = Parallel(n_jobs=30)(delayed(fillData)(table_part[1]) for table_part in tqdm(df.groupby('symbol')))


train_val_data = TrainValTestData(time_series_list=stock_data_list,
                              train_length=1200,  # 1200 trading days for training
                              validate_length=300,  # 150 trading days for validation
                              history_length=30,  # each input contains 30 days of history
                              sample_step=2,  # jump to days forward for each sampling
                              train_val_gap=10,  # leave a 10-day gap between training and validation)
                              batch_size=2000
                              )

train, val,test, dates_info = train_val_data.get(20110630, order="by_date")
# print(dates_info)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = AlphaNetV3(l2=0.001, dropout=0.0,feature = feature_num,save_path='model/model_v3.pt',max_epoch=100,early_stop_rounds=5)
# model = AlphaNetV2(l2=0.001, dropout=0.0,feature = feature_num,save_path='model/model_v2.pt',max_epoch=100,early_stop_rounds=5)
model.to(device)
model.fit(train,val)
model.load_state_dict(torch.load(model.save_path))
pred,labels = model.predict(test)
pred = torch.cat(pred).ravel().cpu().detach().numpy()
labels = torch.cat(labels).ravel().cpu().detach().numpy()
print(np.corrcoef(pred,labels)[0,1])


model1 = AlphaNetV2(l2=0.001, dropout=0.0,feature = feature_num,save_path='model/model_v2.pt',max_epoch=100,early_stop_rounds=5)
model1.to(device)
model1.fit(train,val)
model1.load_state_dict(torch.load(model1.save_path))
pred1,labels1 = model1.predict(test)
pred1 = torch.cat(pred1).ravel().cpu().detach().numpy()
labels1 = torch.cat(labels1).ravel().cpu().detach().numpy()
print(np.corrcoef(pred1,labels1)[0,1])

# os._exit(0)

# model(train..batch(500).cache(),
#           validation_data=val.batch(500).cache(),
#           epochs=100)
