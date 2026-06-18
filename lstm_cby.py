# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:38:30 2026

@author: 1
"""
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from matplotlib import pyplot



# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
 
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
            
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 1、加载数据
dataset = read_csv('power.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')

# 2、数据归一化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
scaled = scaler.fit_transform(values)

# 3、制作数据
# 变成样本数×特征数的形式
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)
# print(reframed.head())

# 将上面的数据分为训练数据 和 测试数据
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 将训练数据 和 测试数据 的输入和输出分开
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# 将 训练数据 和 测试数据的输入 变成 样本×时序×特征 的形式 reshape 一下
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 4、建立神经网络
# 建立神经网络模型
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

print(model.summary())

# 规定损失函数和优化器
model.compile(loss='mae', optimizer='adam')

# 训练网络
history = model.fit(train_X, train_y, 
                    epochs=50, batch_size=72, 
                    validation_data=(test_X, test_y), 
                    verbose=2,
                    shuffle=False)

# 5、训练完
# 画图
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 做预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# print('跑完了')












