import pandas as pd
import talib as tb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from jqdatasdk import *
import jqdatasdk as jq

# 利用聚宽api获取股票数据
jq.auth('18353353680', 'Cxk8113828')
data = jq.get_price(security="399107.XSHE", end_date='2022-7-1', count=5000)

# 清洗数据，加一列涨跌幅
data = data.dropna(axis=0, how='any')
data['mov_percent'] = [data['close'][i] / data['open'][i] - 1 for i in range(len(data))]

# 加一列涨跌趋势
data['movement'] = [int(i > 0) for i in data['mov_percent']]

# 生成technical indicator
EMA_5 = tb.EMA(data.get('close').values, 5)
for i in range(len(data) - 1, -1, -1):
    if np.isnan(EMA_5[i - 1]):
        break
    EMA_5[i] = EMA_5[i] / EMA_5[i - 1] - 1
data['EMA-5'] = EMA_5
data['ROC-5'] = tb.ROC(data.get('close').values, 5)
data['RSI'] = tb.RSI(data.get('close').values)
data['MI'] = tb.MOM(data.get('close').values)
data = data[15:]

# 数据标准化
raw_data = data.get(['mov_percent', 'EMA-5', 'ROC-5', 'RSI']).values
Scaler = MinMaxScaler().fit(raw_data)
data[['mov_percent', 'EMA-5', 'ROC-5', 'RSI']] = Scaler.transform(raw_data)
data.to_csv("深证A指_processed.csv")
