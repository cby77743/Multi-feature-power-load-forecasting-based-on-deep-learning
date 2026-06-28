import pandas
import numpy as np
import torch.nn as nn
from pandas import DataFrame
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from pandas import concat                         

from torch.utils.data import TensorDataset, DataLoader
import torch

def series_to_superised(data, n_in, n_out, dropnan = True):
    """"把时间序列转化为监督学习数据"""
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names =[], []
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]
        
    for i in range(n_out):
        cols.append(df.shift(-i))
        suffix = "(t)" if i == 0 else f"(t+{i})"
        names += [f"var{j + 1}{suffix}" for j in range(n_vars)]
                
    result = concat(cols, axis=1)
    result.columns = names
    if dropnan:
        result.dropna(inplace=True)
    return result

class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=64, batch_first = True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :]).squeeze(-1)
        

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0 
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item() * targets.size(0)
            sample_count += targets.size(0)
    return total_loss / sample_count


def predict( model, loader, device   ):
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(outputs)



# 1、导入数据
dataset = pandas.read_csv('power.csv', header=0, index_col=0)
data_values = dataset.values

data_values = dataset.to_numpy(dtype=np.float32)

# 2、归一化
train_hours = 365*24
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(data_values[:train_hours, :])
scaled = scaler.transform(data_values)

# 3、制作训练数据
reframed = series_to_superised(scaled, n_in=1, n_out=1)
reframed = reframed.values
train = reframed[:train_hours, :5]
test = reframed[train_hours+1:, :5]

train_x = train[:, :4].reshape(train.shape[0], 1, train[:,:4].shape[1])
train_y = train[:, 4]
test_x = test[:, :4].reshape(test.shape[0], 1, test[:, :4].shape[1])
test_y = test[:, 4]

train_dataset = TensorDataset( torch.from_numpy(train_x), torch.from_numpy(train_y))
test_dataset = TensorDataset( torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 72
train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=False )
test_loader = DataLoader( test_dataset, batch_size = batch_size, shuffle=False )


# 4、建立模型 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
model = LoadForecastLSTM(input_size = train_x.shape[2]).to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters())

print(model)
print(f"训练设备：{device}，训练样本：{len(train_dataset)}，测试样本：{len(test_dataset)}")

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
            
            total_loss += loss.item() * targets.size(0)
            
    train_loss = total_loss / len(train_dataset)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
            
    print(f"Epoch {epoch + 1:02d}/{epochs} - loss:{train_loss:.6f} - val_loss:{test_loss:.6f}" )
            
            

predicted_scaled = predict(model, test_loader, device)
    
power_min = scaler.data_min_[0]
power_range = scaler.data_range_[0]

predicted_power = power_min + predicted_scaled * power_range
actual_power = test_y * power_range + power_min

rmse = float(np.sqrt(np.mean((actual_power - predicted_power) ** 2)))
nonzero = actual_power != 0 


# TensorDataset 
# DataLoader 











