# -*- coding: utf-8 -*-
"""使用多特征 LSTM 预测下一小时的电力负荷。"""

import argparse

import numpy as np
import torch
from matplotlib import pyplot
from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """把时间序列转换为监督学习数据。"""
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
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item() * targets.size(0)
            sample_count += targets.size(0)
    return total_loss / sample_count


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def main(epochs=50, batch_size=72, show_plot=False):
    # 1. 加载数据
    dataset = read_csv("power.csv", header=0, index_col=0)
    required_columns = ["power", "temperature", "humidity", "speed"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"power.csv 缺少字段: {', '.join(missing_columns)}")

    raw_values = dataset[required_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(raw_values).all():
        raise ValueError("power.csv 中存在空值或非数值数据")

    # 2. 归一化。缩放器只使用训练区间拟合，防止测试集信息泄漏。
    n_train_hours = min(365 * 24, len(raw_values) - 1)
    if n_train_hours < 1:
        raise ValueError("数据量不足，至少需要两行数据")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(raw_values[: n_train_hours + 1])
    scaled = scaler.transform(raw_values)

    # 使用 t-1 时刻的全部特征预测 t 时刻的 power。
    reframed = series_to_supervised(scaled, n_in=1, n_out=1)
    input_columns = [f"var{i}(t-1)" for i in range(1, len(required_columns) + 1)]
    target_column = "var1(t)"
    supervised = reframed[input_columns + [target_column]].to_numpy(dtype=np.float32)

    train = supervised[:n_train_hours]
    test = supervised[n_train_hours:]
    if len(test) == 0:
        raise ValueError("没有可用的测试数据，请增加数据量或缩短训练区间")

    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # LSTM 输入形状：[样本数, 时间步, 特征数]
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. 建立并训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = LoadForecastLSTM(input_size=train_x.shape[2]).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    print(model)
    print(f"训练设备: {device}，训练样本: {len(train_dataset)}，测试样本: {len(test_dataset)}")

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        train_loss = total_loss / len(train_dataset)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs} - loss: {train_loss:.6f} - val_loss: {test_loss:.6f}")

    # 4. 预测并还原为原始功率单位
    predicted_scaled = predict(model, test_loader, device)
    power_min = scaler.data_min_[0]
    power_range = scaler.data_range_[0]
    predicted_power = predicted_scaled * power_range + power_min
    actual_power = test_y * power_range + power_min

    rmse = float(np.sqrt(np.mean((actual_power - predicted_power) ** 2)))
    nonzero = actual_power != 0
    mape = float(np.mean(np.abs((predicted_power[nonzero] - actual_power[nonzero]) / actual_power[nonzero])))
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAPE: {mape:.3%}")

    pyplot.plot(train_losses, label="train")
    pyplot.plot(test_losses, label="test")
    pyplot.xlabel("epoch")
    pyplot.ylabel("MAE")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig("lstm_cby_pytorch_loss.png", dpi=150)
    if show_plot:
        pyplot.show()
    else:
        pyplot.close()
    print("损失曲线已保存到 lstm_cby_pytorch_loss.png，程序正常结束。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数，默认 50")
    parser.add_argument("--batch-size", type=int, default=72, help="批大小，默认 72")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="训练完成后显示损失曲线窗口；默认保存图片并自动退出",
    )
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, show_plot=args.show_plot)
