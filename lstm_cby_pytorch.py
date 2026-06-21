

import numpy as np
import torch
from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset


from torch import nn

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """把时间序列转化为监督学习数据"""
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

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

    print("def 跑完了")




class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
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
            inputs, targets =  inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item() * targets.size(0)
            sample_count += targets.size(0)
    return total_loss / sample_count







# 1、加载数据
dataset = read_csv("power.csv", header=0, index_col=0)
required_columns = ["power", "temperature", "humidity", "speed"]


missing_columns = [column for column in required_columns if column not in dataset.columns]
if missing_columns:
    raise ValueError(f"power.csv 缺少字段: {', '.join(missing_columns)}")

raw_values = dataset[required_columns].to_numpy(dtype=np.float32)

if not np.isfinite(raw_values).all():
    raise ValueError("power.csv 中存在空值或非数值数据")

# 2、归一化
n_train_hours = min(365 * 24, len(raw_values) - 1)
if n_train_hours < 1:
    raise ValueError("数据量不足，至少需要两行数据")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(raw_values[: n_train_hours + 1])
scaled = scaler.transform(raw_values)

# 3、制作训练数据
reframed = series_to_supervised(scaled, n_in=1, n_out=1)
reframed_value = reframed.values

train_hours = 365*24
train = reframed_value[:train_hours, :5]
test = reframed_value[train_hours:, :5]

train_x = train[:, :4].reshape(train.shape[0], 1, train[:, :4].shape[1])
train_y = train[:, 4]
test_x = test[:, :4].reshape(test.shape[0], 1, test[:, :4].shape[1])
test_y = test[:, 4]


batch_size = 72

train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 4、建立 并且 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
model = LoadForecastLSTM(input_size=train_x.shape[2]).to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters())

print(model)
print(f"训练设备: {device}，训练样本: {len(train_dataset)}，测试样本: {len(test_dataset)}")

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
    print(f"Epoch {epoch + 1:02d}/{epochs} - loss: {train_loss:.6f} - val_loss: {test_loss:.6f}")





print(len(raw_values))

print("跑完了")




# if __name__ == "__main__":
#    main()



