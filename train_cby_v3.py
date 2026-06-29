# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:04:37 2026

@author: 1
"""

from pandas import read_csv
from pandas import DataFrame, concat
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# from torch.utils.data import TensorDataset, DataLoader

import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn as nn

def series_to_super(data, n_in, n_out):
    cols, names =[], []
    df = DataFrame(data)
    for i in range(n_in, 0, -1):
        cols.append( df.shift(i) )
        names += [ f"vars(t-{i})({j+1})" for j in range(data.shape[1]) ]
    
    for i in range(n_out):
        cols.append( df.shift(-i) )
        sufff = "t" if i == 0 else f"t+{i}" 
        names += [ f"vars({sufff})({j+1})" for j in range(data.shape[1]) ]

    results = concat(cols, axis=1)
    results.columns = names
    
    results.dropna(inplace = True)
    return results

class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size = 64 ):
        super().__init__()
        self.lstm = nn.LSTM( input_size=input_size, hidden_size = 64, batch_first = True )
        self.output = nn.Linear( hidden_size, 1 )
        
    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :]).squeeze(-1)

        
# 1、导入数据
dataset = read_csv('power.csv', header = 0, index_col = 0)

dataset_value = dataset.to_numpy(dtype = np.float32)


# 2、将数据归一化
train_hours = 24*365
scaler = MinMaxScaler( feature_range = (0, 1) )
scaler.fit( dataset_value[:train_hours, :])
scaled = scaler.transform(dataset_value)

# 3、制作训练数据

reformed = series_to_super(scaled, n_in = 1, n_out = 1)
reformed_value = reformed.to_numpy(dtype=np.float32)

train_data = reformed_value[:train_hours, :5]
test_data = reformed_value[train_hours+1:, :5]

train_x = train_data[:, :4].reshape(train_data.shape[0], 1, train_data[:, :4].shape[1])
train_y = train_data[:, 4]
test_x = test_data[:, :4].reshape(test_data.shape[0], 1, test_data[:,:4].shape[1])
test_y = test_data[:, 4]

train_tensor = TensorDataset( torch.from_numpy(train_x), torch.from_numpy(train_y) )
test_tensor = TensorDataset( torch.from_numpy(test_x), torch.from_numpy(test_y) )

batch_size = 72
train_loader = DataLoader( train_tensor, batch_size = batch_size, shuffle = False )
test_loader = DataLoader( test_tensor, batch_size = batch_size, shuffle = False )


"""
# 4、建立模型  训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
torch.manual_seed(42)
model = LoadForecastLSTM( input_size = train_x.shape[2] ).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

print(model)
print( f"训练设备：{device}, 训练样本：{len(train_data)}, 测试样本：{len(test_data)}"  )

train_losses, test_losses = [], []

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # 前向传播，算误差
        loss = loss_fn(model(inputs), targets)

        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()*targets.size(0)

    train_loss = total_loss / len(train_data)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print( f"第{epoch+1}/{epochs}次训练，训练数据损失{train_loss:.6f}，测试数据损失{test_loss:.6f}" )
"""


# 4、建立模型 训练模型

"""
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0 
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)   
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item()*targets.size(0)
            sample_count += targets.size(0)
        return total_loss / sample_count
"""

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0 
    sample_count = 0 
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item()*targets.size(0)
            sample_count += targets.size(0)
    return total_loss / sample_count


device = torch.device( "cuda" if torch.cuda.is_available else "cpu"  )
torch.manual_seed(42)
model = LoadForecastLSTM(input_size = train_x.shape[2]).to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

print(model)
print( f"训练设备{device}, 训练数据样本{len(train_data)}, 测试数据样本{len(test_data)}"  )

train_loss_a=[]
test_loss_a=[]
epochs = 50 
for i in range(epochs):
    train_losses = 0.0
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        
        loss.backward()
        optimizer.step()
        train_losses += loss.item()*targets.size(0)
    train_loss = train_losses / len(train_data)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    train_loss_a.append(train_loss)    
    test_loss_a.append(test_loss)
    print(f"第{i+1:02d}次训练，训练损失：{train_loss:.6f},测试损失:{test_loss:.6f} "   )

"""
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
torch.manual_seed(42)

model = LoadForecastLSTM(input_size = train_x.shape[2]).to(device)

loss_fn = nn.L1Loss()
optimizer =  torch.opti.Adam()

print(model)
"""


"""
train_losses = []
test_losses = []

epochs = 50 
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*targets.size(0)
    
    train_loss = total_loss / len( train_y )
    test_loss = evaluate(model, test_loader, loss_fn, device) 
    train_losses.append(train_loss)
    test_losses.append(test_loss)
  """ 
    





        

   
    
    
    
    








