from pandas import read_csv
from datetime import datetime


# 读取原始数据
data_set = read_csv('raw_data.csv', index_col=0)

# 取出数值
data_value = data_set.values

# 加入数据标签
data_set.columns = ['power', 'temperature', 'humidity', 'speed']

# 去掉前24行
data_set = data_set[24:]

# 输出得到的数据
print(data_set.head(5))

data_set.to_csv('power.csv')

print('跑完了')

# summarize first 5 rows
# print(dataset.head(5))
# save to file
# dataset.to_csv('power.csv')




