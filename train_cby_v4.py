# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:32:58 2026

@author: 1
"""

import numpy as np
import torch
from pandas import read_csv, DataFrame, concat

import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


def series_to_sup( data, n_in, n_out ):
    cols = []
    names = []
    df = DataFrame(data)    # 这个写的不熟练
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f" var(t-{i})({j+1})" for j in range(data.shape[1])]
    for i in range(n_out):
        cols.append(df.shift(-i))
        names += [f" var(t+{i})({j+1}))" for j in range(data.shape[1])]
        
    result = concat( cols, axis = 1 )
    result.dropna(inplace = True )  # 这个不熟练
    result.columns = names
    
    return result
    

class LoadForecastLSTM(nn.Module):      # 这整个类写的就不好，不熟练 不会写
    def __init__(self, input_size, hidden_size = 64):  
        super().__init__()
        self.lstm = nn.LSTM( input_size = input_size, hidden_size = hidden_size, batch_first = True)  # 这两行创建 网络 不会写
        self.output = nn.Linear( hidden_size, 1 )  # input_size = hidden_size 这个地方为什么不能这样写呢
    
    def forward(self, x):  # 这个函数也不会写
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :]).squeeze(-1)
   
      
    
def evaluate(model, data, loss_fn, device):
    model.eval()
    test_losses = 0.0
    sample_count = 0
    with torch.no_grad():  # 这个容易忘
        for inputs, targets in data:
            inputs, targets = inputs.to(device), targets.to(device)
        
            test_losses += loss_fn(model(inputs), targets).item()*targets.size(0)
            sample_count += targets.size(0)
    return test_losses / sample_count
    
def predict(model, data, device):  # 这个还不会写
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in data:
            outputs.append( model(inputs.to(device)).cpu().numpy())
    return np.concatenate(outputs)
 
    
# 1、导入数据
dataset = read_csv('power.csv', header = 0, index_col = 0)
data_value = dataset.to_numpy(dtype = np.float32)

# 2、数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))  # 这个0和1在写的时候 为什么是小括号不是方括号呢
train_hours = 24*365
scaler.fit( data_value[ :train_hours, : ] )
scaled =  scaler.transform(data_value)

# 3、制作训练数据
reformed = series_to_sup( scaled, n_in = 1, n_out = 1 )
reformed = reformed.to_numpy(dtype = np.float32)  # 这行代码忘写了，忘了转化为数值了后面

train = reformed[ :train_hours, :5 ]
test = reformed[ train_hours+1:, :5 ]
train_x = train[ :, :4 ].reshape( train.shape[0], 1, train[:, :4].shape[1])
train_y = train[ :, 4]
test_x = test[:, :4].reshape( test.shape[0], 1, test[:, :4].shape[1] )
test_y = test[:, 4]

train_tensor = TensorDataset( torch.from_numpy(train_x), torch.from_numpy(train_y) ) #这个地方没有写torch
test_tensor = TensorDataset( torch.from_numpy(test_x), torch.from_numpy(test_y) )
batch_size = 72
train_loader = DataLoader( train_tensor, batch_size, shuffle = False )  # 这个不熟练 忘记写 batch_size,不太记得应该写什么选项
test_loader = DataLoader( test_tensor, batch_size, shuffle = False )

# 4、建立网络模型，训练
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

model = LoadForecastLSTM( input_size=train_x.shape[2] ).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())  # 这个地方不知道写括号里面的参数

print(model)
print("  训练样本数量   测试样本数量  "    )


epochs = 50 
total_loss = 0.0 
train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0 
    sample_count = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn( model(inputs), targets )
        loss.backward()  # 这个地方不会写 我写的是loss.backforward 
        optimizer.step() # 这个地方不会写  我写的是 optimizer.loss
        
        total_loss += loss.item()*targets.size(0)
        sample_count += targets.size(0)

    train_loss = total_loss / sample_count
    test_loss = evaluate( model, test_loader, loss_fn, device )         
    train_losses.append(train_loss)
    test_losses.append(test_loss)
        
    print(f"第{epoch+1:02d}/{epochs}训练, 训练损失：{train_loss:.6f}，测试损失：{test_loss:.6f} ")
    
"""
#4、预测、反归一化、算评价指标  # 下面这个还不会写
predict_scaled = predict( model, test_loader, device)
power_min = scaler.data_min_[0]
power_range = scaler.data_range_[0]

predicted_power = predict_scaled * power_range + power_min
acture_power = test_y * power_range + power_min
"""

def predict(model, data, device):
    model.eval()
    result= []
    with torch.no_grad:
        for inputs, _ in data:
            result.append( model(inputs.to(device)).cpu().to_numpy() )
    return result.concatenate(result)
    

predict_scaled = predict( model, test_loader, device )
power_min = scaler.data_min_[0]
power_range = scaler.data_range_[0]
predict_power = predict_scaled*power_range + power_min
acture_power = test_y*power_range + power_min

rmse = float( np.sqrt(np.mean((acture_power - predict_power) ** 2)) )

rmse = float( np.sqrt(np.mean((acture_power - predict_power) ** 2)) )
nonzero = acture_power != 0
mape = float(np.mean(np.abs(( predict_power[nonzero] - acture_power[nonzero] ))/acture_power[nonzero]))

print( f"Test RMSE: {rmse:.3f}" )
print( f"Test MAPE: {mape:.3%}" )

















