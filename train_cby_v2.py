# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:05:18 2026

@author: byche
"""

import pandas
import numpy as np
import torch

from pandas import DataFrame
from pandas import read_csv
from pandas import concat

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler 


"""
def series_to_superised( data, n_in, n_out ):
    
    n_var = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [],[]
    names_v2 = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(data.shape[1])]
        
    for i in range(n_out):
        cols.append(df.shift(-i))
        suffix = "(t)" if i == 0 else f"(t+{i}"
        names += [f"var{j + 1}{suffix}" for j in range(n_var)]
        
    result = concat(cols,axis = 1)
    result.columns = names
    result.dropna(inplace=True)
"""       
    

def series_to_superised( data, n_in = 1, n_out = 1):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    cols, names = [], []
    df_data = DataFrame(data)
    
    for i in range(n_in, 0, -1):
        cols.append(df_data.shift(i))
        names += [ f" vars(t - {i}) ({j+1}) " for j in range(n_vars) ]
    
    for i in range(n_out):
        cols.append(df_data.shift(-i))
        suffff = "t" if i == 0 else f"t+{i}"
        names += [f" var({suffff})({j+1})" for j in range(n_vars) ]

    result = concat(cols, axis=1)
    result.columns = names
    result.dropna(inplace=True)
    return result
    
    
class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size = 64):
        super().__init__()
        self.lstm = nn.LSTM( input_size = input_size, hidden_size=64, batch_first = True)
        self.output = nn.Linear(hidden_size, 1)



# 1、读取数据
data_set = read_csv("power.csv", header = 0, index_col = 0 )

data_values = data_set.to_numpy(dtype = np.float32)

# 2、归一化
train_hours = 24*365
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(data_values[:train_hours, :])

scaled = scaler.transform(data_values)


# 3、制作训练数据
reframed = series_to_superised(scaled, n_in = 1, n_out = 1 )

reframed_value = reframed.to_numpy(dtype = np.float32)

train_data = reframed_value[ :train_hours, :5]
test_data = reframed_value[  train_hours:,:5 ]

train_x = train_data[:, :4].reshape(train_data.shape[0], 1, train_data[:, :4].shape[1] )
train_y = train_data[:, 4]
test_x = test_data[:, :4].reshape(test_data.shape[0], 1, test_data[:, :4].shape[1])
test_y = test_data[:, 4]

# DataLoader
# TensorData

train_tensor = TensorDataset( torch.from_numpy(train_x), torch.from_numpy(train_y) )
test_tensor = TensorDataset( torch.from_numpy(test_x), torch.from_numpy(test_y)  )

batch_size = 72

train_loader = DataLoader( train_tensor, batch_size = batch_size, shuffle=False )
test_loader = DataLoader( test_tensor, batch_size = batch_size, shuffle=False )


# 4、建立模型，训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)























