# 多特征电力负荷预测 — 学习笔记

---

## 目录

- [一、项目概览](#一项目概览)
- [二、Python 基础知识](#二python-基础知识)
  - [内置类型 vs 第三方库类型](#21-内置类型-vs-第三方库类型)
  - [pandas 是什么、datetime 是什么](#22-pandas-是什么datetime-是什么)
  - [from ... import ... 三种写法](#23-from--import--三种写法)
  - [index_col=0 的作用](#24-index_col0-的作用)
  - [Python 代码组织层级：Library → Module → Class → Object → Method/Attribute/Function](#25-python-代码组织层级library--module--class--object--methodattributefunction)
- [三、VS Code 使用](#三vs-code-使用)
  - [终端 Debug 命令的含义](#31-终端-debug-命令的含义)
  - [FileNotFoundError 的原因](#32-filenotfounderror-的原因)
  - [VS Code vs Spyder vs Jupyter Notebook](#33-vs-code-vs-spyder-vs-jupyter-notebook)
- [四、怎么看 Python 报错](#四怎么看-python-报错)
- [五、LSTM 预测程序写作框架](#五lstm-预测程序写作框架)
- [六、def 函数 vs 主程序](#六def-函数-vs-主程序)
  - [执行时机](#61-执行时机)
  - [什么时候抽成 def：三问判断法](#62-什么时候抽成-def三问判断法)
- [七、开源软件生态](#七开源软件生态)
- [八、各组件详解](#八各组件详解)
- [九、LSTM 底层原理](#九lstm-底层原理)

---

## 一、项目概览

### 这个项目是做什么的？

这是一个**基于深度学习的多特征电力负荷预测**项目。核心目标：利用历史电力负荷数据以及天气特征（温度、湿度、风速），通过 LSTM 深度学习模型预测未来的电力负荷。

项目规划了 5 个版本的演进方案（V1～V5），当前 `power_load_forecasting_V1/` 文件夹只实现了 **V1 版本**：利用**上一时刻**的电力负荷 + 温度 + 湿度 + 风速 → 预测**此刻**的电力负荷。

| 版本 | 预测策略 |
|------|----------|
| **V1** | 利用**上一时刻**的 电力负荷 + 温度/湿度/风速 → 预测**此刻**负荷 |
| **V2** | 利用**上一时刻**的电力负荷 + **此刻**的温度/湿度/风速 → 预测**此刻**负荷 |
| **V3** | 利用**过去若干时刻**的电力负荷 + 温度/湿度/风速 → 预测**此刻**负荷 |
| **V4** | 按**季节**（夏/冬/过渡季）分开建模预测 |
| **V5** | 在季节划分基础上，再按**天/周/月**周期细分 |

### 程序结构

```
├── README.md                          # 项目总体说明（5版本规划）
├── raw_data/data.xlsx                 # 原始 Excel 数据
└── power_load_forecasting_V1/
    ├── raw_data.csv                    # 原始 CSV 数据
    ├── data_processing.py              # 数据预处理（清洗、命名列、移除前24小时）
    ├── data_show.py                    # 数据可视化（各特征时序图）
    ├── power.csv                       # 预处理后输出
    └── lstm_predict.py                 # LSTM模型训练与预测（核心）
```

### 运行顺序

```
raw_data.csv  (原始数据)
     ↓  data_processing.py  ← 第一步
power.csv     (清洗后的数据)
     ↓  lstm_predict.py    ← 第二步
模型训练 + 预测结果
```

### 三个 Python 文件职责

| 文件 | 功能 |
|------|------|
| `data_processing.py` | 读取 CSV，命名 4 列（power/temperature/humidity/speed），丢弃前 24 小时，输出 `power.csv` |
| `data_show.py` | 绘制各特征列的时序折线图，用于探索性数据分析 |
| `lstm_predict.py` | **核心**：`series_to_supervised()` 构造监督样本 → MinMaxScaler 归一化 → 前 365 天为训练集 → **LSTM(64单元) + Dense(1)** 模型训练 → RMSE 和 MAPE 评估 |

### 模型架构

```
LSTM(64 units) → Dense(1)
损失函数: MAE | 优化器: Adam | Epochs: 50 | Batch: 72
输入: t-1 时刻的 [power, temperature, humidity, speed] (4特征, 1时间步)
输出: t 时刻的 power
```

---

## 二、Python 基础知识

### 2.1 内置类型 vs 第三方库类型

Python 教程里教的都是"内置类型"——装 Python 的时候就有的东西：

```python
# Python 内置类型（不需要安装任何东西就能用）
a = [1, 2, 3]           # 列表 list
b = {"name": "张三"}     # 字典 dict
c = (1, 2)              # 元组 tuple
d = {1, 2, 3}           # 集合 set
e = 3.14                # 浮点数 float
f = "hello"             # 字符串 str
g = True                # 布尔值 bool
```

**DataFrame 和 ndarray 不是 Python 自带的，是别人写的"扩展包"里的。**

```
Python 安装后就有的（内置）         需要 pip install 安装的（第三方库）
─────────────────────────────     ─────────────────────────────
list    [1, 2, 3]                 numpy 库  → ndarray（多维数组）
dict    {"a": 1}                  pandas 库 → DataFrame（表格）
tuple   (1, 2)                    keras 库  → LSTM 模型
set     {1, 2}                    matplotlib → 画图
int/float/str/bool                ...上百个其他库
```

**DataFrame** = "Python 里的 Excel"——有行标签、列名、纯数据的表格。

```
DataFrame
  ├── .values   →  ndarray（纯数字，给模型吃）
  ├── .columns  →  列名（知道每列是什么意思）
  └── .index    →  行标签（知道每行是什么意思）
```

**ndarray** = C 语言写的"超级列表"——外表像列表，但计算速度快 100 倍：

```python
import numpy as np
arr = np.array([[796, 0.72, 0.8, 0.23],
                [758, 0.72, 0.8, 0.23]])

# 可以对整个数组做数学运算（列表做不到！）
arr * 2        # 每个数字 ×2，一行代码
arr.mean()     # 求所有数字的平均值，一行代码
```

| | Python 列表 | numpy ndarray |
|---|---|---|
| 速度 | 慢（Python 循环） | 快（C 语言底层）|
| 数学运算 | `[x*2 for x in lst]` | `arr * 2` |
| 几万行数据 | 卡顿 | 流畅 |
| 来源 | Python 自带 | `pip install numpy` |

### 2.2 pandas 是什么、datetime 是什么

**pandas** = 专门处理表格数据的"外挂工具箱"，就像"Python 里的 Excel 操作器"。

| 你想做的事 | Excel 里怎么做 | pandas 里怎么做 |
|-----------|-------------|--------------|
| 打开 CSV | 双击文件 | `read_csv('文件.csv')` |
| 删前 24 行 | 选中 → 右键删除 | `df[24:]` |
| 改列名 | 双击表头改 | `df.columns = [...]` |
| 另存为 | Ctrl+S | `df.to_csv('新文件.csv')` |

名字来源：**pan**el **da**ta（面板数据）的缩写，不是 🐼。

**datetime** = Python 自带的"时间处理器"（安装 Python 就有，不需要 pip install）。在 `data_processing.py` 里它被 import 了但实际上没用到，可以安全删掉。

### 2.3 from ... import ... 三种写法

```python
# 方式 1：只拿一个工具
from pandas import read_csv

read_csv(...)              # ✅ 能用
DataFrame(...)             # ❌ 报错！没导入

# 方式 2：拿整个工具箱进来
import pandas as pd

pd.read_csv(...)           # ✅ 能用（要加 pd. 前缀）
pd.DataFrame(...)          # ✅ 能用
# pandas 里几百个函数全部能用，但前面都要写 pd.

# 方式 3：一次拿多个工具
from pandas import read_csv, DataFrame

read_csv(...)              # ✅ 能用
DataFrame(...)             # ✅ 能用
```

比喻：
- `import pandas` → "我把整个五金店搬过来了"，每个工具用 `pd.某某某()` 拿
- `from pandas import read_csv` → "我只从五金店借了一把螺丝刀"，只能用这个

### 2.4 index_col=0 的作用

`index_col=0` 把 CSV 的第 0 列（time）从"数据列"中抽走，变成**行标签**（index）。

**为什么这样做？** 因为 LSTM 模型只吃数字，不吃文字。"2013/9/1 0:00" 这个时间字符串是文字，如果留在列里，`dataset.values` 会包含它，模型吃进去就报错。

```
raw_data.csv 原始有 5 列:
  time             power  temperature  humidity  speed
  2013/9/1 0:00    796.67 0.72         0.80      0.23

index_col=0 之后:
  dataset.index   →  [2013/9/1 0:00, ...]   ← time 变成行标签
  dataset.columns →  ['power', 'temperature', 'humidity', 'speed']  ← 只剩 4 个数据列
  dataset.values  →  [796.67, 0.72, 0.80, 0.23]  ← 纯数字，没有 time
```

**所以 `dataset.columns = ['power', ...]` 只给 4 列命名，time 不在其中——因为 time 已经不是列了，是行标签。**

DataFrame 设计的精妙之处：把"元信息"和"数据"分开——
- `.index` → 行标签（给人看）
- `.columns` → 列名（给人看）
- `.values` → 纯数字（给模型吃）

### 2.5 Python 代码组织层级：Library → Module → Class → Object → Method/Attribute/Function

#### 2.5.1 层级总览

Python 代码是按层级组织的，从大到小依次是：

```
Library (库)
  └── Module / Submodule (模块/子模块)
       └── Class (类)  ←── 用它造出──  Object / Instance (对象/实例)
            │                              │
            ├── Method (方法) ← 对象自带的函数  调用: df.dropna()
            └── Attribute (属性) ← 对象自带的数据  访问: df.columns
       
  └── Function (独立函数) ← 不属于某个对象，直接调  调用: concat()
```

| 层级 | English | 中文 | 一句话解释 | 例子 |
|:--|:--|:--|:--|:--|
| 1 | **Library** / Package | 库 / 包 | 一个工具箱，一堆代码的集合 | `pandas`、`numpy`、`keras` |
| 2 | **Module** / Submodule | 模块 / 子模块 | 库里面的"文件夹"，按功能分组 | `keras.models`、`pandas` |
| 3 | **Class** | 类 | 一个"模具"/"蓝图"，用来创建对象 | `DataFrame`、`Sequential`、`LSTM` |
| 4 | **Object** / Instance | 对象 / 实例 | 用类造出来的"实体" | `df`、`model`、`scaler` |
| 5A | **Method** | 方法 | 对象自带的函数，调用：`对象.方法()` | `df.dropna()`、`model.fit()` |
| 5B | **Attribute** / Property | 属性 | 对象自带的数据，访问：`对象.属性` | `df.columns`、`data.shape` |
| — | **Function** | 独立函数 | 不依附于对象的函数，直接调：`函数()` | `concat()`、`read_csv()` |
| — | **Parameter** / Argument | 参数 | 传给函数的输入值 | `n_in=1`、`axis=1`、`inplace=True` |

#### 2.5.2 各层级详解

**① Library（库）**：`pip install` 安装的东西。一堆 Python 文件的集合，提供特定领域的功能。

```python
# 先安装：pip install pandas
import pandas         # 把整个 pandas 库引入
```

**② Module / Submodule（模块/子模块）**：库内部的"文件夹"，按功能把代码分门别类。

```python
# keras 库的目录结构（简化）：
keras
├── models       ← 子模块：装模型定义相关（Sequential, Model）
├── layers       ← 子模块：装各种网络层（LSTM, Dense, Dropout）
├── optimizers   ← 子模块：装优化器（Adam, SGD）
├── losses       ← 子模块：装损失函数（mae, mse）
└── ...

# 从不同子模块拿不同东西：
from keras.models import Sequential      # 从 models 子模块拿 Sequential 类
from keras.layers import LSTM, Dense      # 从 layers 子模块拿 LSTM、Dense 类
```

> `from A.B import C` 的意思是：从 **A库** 的 **B子模块** 里拿 **C**。

**③ Class（类）**：一个"模具"或"蓝图"。类本身不干活，它定义了"造出来的是什么东西、能做什么"。

```python
# 这些全是类（Class）：
DataFrame      # pandas 里的表格模具
Sequential     # Keras 里的顺序模型模具
LSTM           # Keras 里的 LSTM 层模具
Dense          # Keras 里的全连接层模具
MinMaxScaler   # sklearn 里的归一化模具
```

类不能直接用，得先"造对象"：

```python
Sequential.add(...)    # ❌ 报错！类不能直接调用方法
model = Sequential()   # ✅ 先造一个对象 model
model.add(...)         # ✅ 然后调用对象的方法
```

**类比：** 类 = 饼干模具，对象 = 用模具压出来的饼干。模具本身不能吃，饼干才能吃。

**④ Object / Instance（对象/实例）**：用类造出来的"实体"，在内存里真实存在，存储数据。

```python
df = DataFrame(data)       # df 是对象（用 DataFrame 模具造的）
model = Sequential()       # model 是对象（用 Sequential 模具造的）
scaler = MinMaxScaler()    # scaler 是对象（用 MinMaxScaler 模具造的）

# 同一个类可以造无数个对象：
df1 = DataFrame({'a': [1, 2]})
df2 = DataFrame({'a': [3, 4]})
# df1 和 df2 是两个独立的对象，互不影响
```

**⑤ Method（方法）**：对象自带的函数。调用格式是 `对象.方法名()`。

```python
df = DataFrame(data)
df.dropna()          # dropna 是 DataFrame 对象的方法
df.head()            # head 是方法，显示前5行
df.to_csv(...)       # to_csv 是方法，保存到文件

model = Sequential()
model.add(LSTM(64))  # add 是 Sequential 对象的方法
model.compile(...)   # compile 是方法
model.fit(...)       # fit 是方法，开始训练
```

**⑤ Attribute（属性）**：对象自带的数据。访问格式是 `对象.属性名`（没有括号！）。

```python
df.columns           # 属性：列名
df.index             # 属性：行标签
df.values            # 属性：纯数字数组
data.shape           # 属性：数组的形状 (行数, 列数)
```

| | Method（方法） | Attribute（属性） |
|:--|:--|:--|
| 是什么 | 对象会做的"动作" | 对象身上的"数据" |
| 调用方式 | `对象.方法()` 有括号 | `对象.属性` 无括号 |
| 例子 | `df.head()` | `df.columns` |

**⑥ Function（独立函数）**：不属于任何对象的函数，直接调用。谁都可以用，不需要"造对象"。

```python
# 这些是独立函数（function），不是方法（method）：
concat([df1, df2], axis=1)     # pandas 里的拼接函数
read_csv('file.csv')           # pandas 里的读文件函数
```

**怎么区分 Function 和 Method？**

```python
# Method：对象.名字()
df.dropna()          # df 是对象，dropna 是它自带的 → Method

# Function：直接调，前面没对象
read_csv(...)        # 前面没有对象 → Function
concat(...)          # 前面没有对象 → Function
```

> **口诀："前面带点的叫 Method（方法），前面不带点的叫 Function（独立函数）。"**

#### 2.5.3 判断要不要 import 的终极口诀

```python
# 口诀：前面没点的 → 一定需要 import
#       前面有点的 → 不需要你 import（对象已经有了）

read_csv(...)        # 前面没点 → 必须 from pandas import read_csv
DataFrame(...)       # 前面没点 → 必须 from pandas import DataFrame
Sequential()         # 前面没点 → 必须 from keras.models import Sequential

df.head()            # 前面有点（df）→ head 不需要单独 import
model.fit(...)       # 前面有点（model）→ fit 不需要单独 import
```

> **Step 1：先看前面有没有点。**
> **Step 2：有点 → 不用管，对象已经有了这个方法。**
> **Step 3：没点 → 必须 import 对应的类或函数。**

#### 2.5.4 你代码里的实际对照表

以 `lstm_predict.py` 为例：

| 代码 | Category | English | 需要 import 吗？ | import 写法 |
|:--|:--|:--|:--|:--|
| `pandas` | Library | library | ✅ | `import pandas` |
| `DataFrame` | Class | class | ✅ | `from pandas import DataFrame` |
| `df = DataFrame(data)` 中的 `df` | Object | object / instance | ❌ 不用 | — |
| `df.shift(1)` | Method | method | ❌ 不用 | — |
| `df.dropna()` | Method | method | ❌ 不用 | — |
| `df.columns` | Attribute | attribute | ❌ 不用 | — |
| `data.shape[1]` | Attribute | attribute | ❌ 不用 | — |
| `concat(cols, axis=1)` | Function | function | ✅ | `from pandas import concat` |
| `read_csv('power.csv')` | Function | function | ✅ | `from pandas import read_csv` |
| `Sequential` | Class | class | ✅ | `from keras.models import Sequential` |
| `LSTM` | Class | class | ✅ | `from keras.layers import LSTM` |
| `Dense` | Class | class | ✅ | `from keras.layers import Dense` |
| `MinMaxScaler` | Class | class | ✅ | `from sklearn.preprocessing import MinMaxScaler` |
| `model = Sequential()` 中的 `model` | Object | object / instance | ❌ 不用 | — |
| `model.fit(...)` | Method | method | ❌ 不用 | — |
| `model.add(...)` | Method | method | ❌ 不用 | — |

#### 2.5.5 `from keras.models import Sequential` 路径拆解

```
from keras.models import Sequential
      ↑      ↑             ↑
    keras   models      Sequential
    库      子模块          类
  (Library) (Module)    (Class)

路径含义：
  keras           → 深度学习库（大楼）
    └── models    → 模型定义子模块（大楼里的一个房间）
         └── Sequential → 顺序堆叠模型类（房间里的一个工具）
```

同理：

```python
from keras.layers import LSTM, Dense
#     ↑       ↑           ↑     ↑
#   keras   layers      LSTM   Dense
#    库      子模块       类     类

from sklearn.preprocessing import MinMaxScaler
#     ↑          ↑               ↑
#  sklearn   preprocessing    MinMaxScaler
#    库         子模块             类
```

#### 2.5.6 记忆口诀总结

| 概念 | 口诀 |
|:--|:--|
| Class vs Object | **"Class 是模具，Object 是实物"** |
| Method vs Function | **"前面带点的叫 Method，前面不带点的叫 Function"** |
| 要不要 import | **"前面没点要 import，前面有点不用管"** |
| Class 不能直接用 | **"模具不能干活，造出实物才能用"** |

### 2.6 Python 三种括号速查

| 括号 | 名称 | 用途 | 例子 |
|---|---|---|---|
| `()` | 小括号/圆括号 | ① 调用函数 ② 元组 ③ 控制运算优先级 | `print("hello")` `(1, 2)` `(1+2)*3` |
| `[]` | 中括号/方括号 | ① 列表 ② 索引取值 ③ 列表推导式 | `[1, 2, 3]` `arr[0]` `[f"..." for j in range(4)]` |
| `{}` | 大括号/花括号 | ① 字典 ② 集合 ③ f-string 占位符 | `{"a": 1}` `{1, 2}` `f"{x}"` |

#### () 小括号 / 圆括号

```python
# ① 调用函数（最常见的用法）
result = max(10, 20)           # 调用 max 函数，传入参数 (10, 20)
scaler = MinMaxScaler()        # 创建对象（类的"构造函数"调用）
df.dropna()                    # 调用对象的方法

# ② 元组 — 不可修改的"列表"
coords = (3, 4)                # 一个坐标元组
rgb = (255, 0, 0)              # 颜色元组

# ③ 控制运算优先级（和数学一样）
ans = (1 + 2) * 3              # 先算括号里的 → 9
ans = 1 + 2 * 3                # 先乘除 → 7
```

#### [] 中括号 / 方括号

```python
# ① 列表 — 装东西的篮子
names = ["张三", "李四", "王五"]     # 一个装字符串的列表
numbers = [1, 2, 3, 4, 5]           # 一个装数字的列表

# ② 索引取值 — 从数组/列表中取第 n 个
arr = np.array([10, 20, 30])
val = arr[0]        # 取第 0 个 → 10
val = arr[1]        # 取第 1 个 → 20

# DataFrame 也可以用 [] 取列
dataset["power"]    # 取 "power" 这一列

# ③ 列表推导式 — 一行代码生成列表
[f"var{j+1}(t-1)" for j in range(4)]
# 结果：["var1(t-1)", "var2(t-1)", "var3(t-1)", "var4(t-1)"]
```

#### {} 大括号 / 花括号

```python
# ① 字典 — "键 → 值" 的映射表
person = {"name": "张三", "age": 20}
person["name"]    # → "张三"
person["age"]     # → 20

# ② 集合 — 不重复的"篓子"
ids = {1, 2, 3, 3}    # 重复的 3 自动去重 → {1, 2, 3}

# ③ f-string 占位符 — {} 里的变量会被替换
name = "张三"
f"我的名字是{name}"     # → "我的名字是张三"
#               ↑↑
#         {} 是占位符，不是字典！
#         只在 f"..." 字符串内部生效
```

#### 和 C/Java 的区别

> Python **没有**"用大括号 `{}` 包住代码块"的语法（那是 C/Java 的习惯）。Python 用 **缩进**（行首空格/Tab）来表示代码块。

```python
# Python — 缩进决定代码块
if x > 0:
    print("正数")      # 缩进 = 属于 if 的代码块
    y = x * 2
print("结束")          # 不缩进 = 不在 if 里

# C/Java — 大括号决定代码块（Python 不是这样的！）
// if (x > 0) {
//     print("正数");
// }
```

#### 记忆口诀

| 括号 | 口诀 |
|---|---|
| `()` | **"小括号调函数"** — 看到 `名字()` 就是在调用 |
| `[]` | **"中括号取元素"** — 看到 `xxx[数字]` 就是在取第几个 |
| `{}` | **"花括号存映射"** — 看到 `{键: 值}` 就是字典 |

---

## 三、VS Code 使用

### 3.1 终端 Debug 命令的含义

当你按 F5 运行 debug 时，终端显示的那一串不是你的代码，是 VS Code 自动生成的启动命令：

```
PS C:\...> (C:\anaconda\Scripts\activate) ; (conda activate base)
PS C:\...> & 'c:\anaconda\python.exe' '...debugpy\launcher' '65328' '--' '...data_processing.py'
```

逐段翻译：

```
&  python.exe    debugpy启动器    端口65328    --    你的程序.py
    ↑               ↑              ↑          ↑        ↑
  "用这个        "开启调试       "通过这个    "下面是  "这是真正
   Python"       模式运行"       端口通信"    真正的   要跑的代码"
                                            程序"
```

- `conda activate base`：VS Code 帮你自动激活 Anaconda 环境
- `python.exe`：用 Anaconda 自带的 Python
- `debugpy\launcher`：VS Code 的调试器，实现断点、单步执行、看变量
- `65328`：通信端口号
- `--`：分隔符，"前面是调试器参数，后面是你真正的程序"

| 运行方式 | 终端显示 | 立足点 |
|----------|---------|--------|
| 终端手动敲 `python xxx.py` | 只有 `python xxx.py` | 你当前 `cd` 到的文件夹 |
| VS Code 按 F5 | 上面那一大串 debugpy 命令 | **项目根目录** |

### 3.2 FileNotFoundError 的原因

```
FileNotFoundError: [Errno 2] No such file or directory: 'raw_data.csv'
```

**程序说："我在当前立足点找不到 raw_data.csv"。**

- 终端手动跑：你 `cd` 进了 `power_load_forecasting_V1`，立足点就是那个文件夹 → raw_data.csv 在旁边 → ✅
- VS Code F5：立足点是**项目根目录** → raw_data.csv 在子文件夹里 → ❌

**解决方案：**
- 方案 A：改路径 `read_csv('power_load_forecasting_V1/raw_data.csv', index_col=0)`
- 方案 B：创建 `.vscode/launch.json`，指定工作目录 `"cwd": "${workspaceFolder}/power_load_forecasting_V1"`

### 3.3 VS Code vs Spyder vs Jupyter Notebook

| | .py 脚本 + VS Code | Spyder | Jupyter Notebook |
|---|---|---|---|
| **形态** | 纯文本文件 + 通用编辑器 | 专为 Python 数据科学设计的 IDE | 网页文档（.ipynb） |
| **运行方式** | F5 一次性跑/断点调试 | F5 运行，右侧有**变量资源管理器** | **逐块运行**，代码+结果+图表嵌在一起 |
| **看 DataFrame** | 断点 → 变量面板 → 右键"查看" | 运行完**自动出现**在变量列表 | 单元格运行完直接显示表格 |
| **画图** | 弹出新窗口 | 弹出新窗口 | 图嵌在代码下面 |
| **写报告** | ❌ | ❌ | ✅ Markdown + 代码混排 |
| **适合** | 正式工程项目、多语言 | 学 Python、数据分析实验 | 数据探索、做实验、写报告 |
| **语言** | 所有语言 | 只 Python | 主要 Python |

**要点：**
- Spyder 看 DataFrame 最直观 → 变量资源管理器自动展示
- Jupyter Notebook 做数据探索最方便 → 逐块运行，输出嵌在下面
- VS Code 写正式项目最好 → 多语言、Git 集成、几万个插件

---

## 四、怎么看 Python 报错

Python 报错格式永远是：

```
错误类型: 错误说明
  File "哪个文件", line 第几行, in 哪个位置
    出错的代码那一行
```

**阅读顺序：从下往上读！** 最下面一行是"最终结论"，往上是"调用链"。

举例：
```
File "...data_processing.py", line 4, in <module>
    dataset = read_csv('raw_data.csv', index_col=0)
FileNotFoundError: ... 'raw_data.csv'
```

→ **第 4 行** `read_csv('raw_data.csv')` → **文件找不到**。

另外那些 conda 相关的报错（`另一个程序正在使用此文件`、`Failed to run 'conda activate base'`）是终端初始化问题，和你的 Python 程序没有关系，可以忽略。

---

## 五、LSTM 预测程序写作框架

写任何一个机器学习程序，都遵循 **"六步流水线"**：

```
① 加载数据       ② 数据预处理      ③ 构造训练格式     ④ 搭建模型       ⑤ 训练+看loss   ⑥ 预测+评估
   CSV → 数组     归一化 0~1        X(输入)/Y(输出)    LSTM 网络         loss 下降图      RMSE, MAPE
```

### 第 ① 步：加载数据

"把 power.csv 读到 Python 里"

```python
dataset = read_csv('power.csv', header=0, index_col=0)
values = dataset.values   # 只取纯数字
```

**输入：** `power.csv`（硬盘上的文件）  
**输出：** `values`（8760行 × 4列 的 numpy 数组）

### 第 ② 步：归一化

"电力负荷是 700~900，温度是 0.6~0.9，尺度不一样——统一缩放到 0~1"

```python
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
scaled = scaler.fit_transform(values)
```

> **为什么要归一化？** 就像考试打分——语文150分，数学100分，不统一成百分比的话语文影响力就比数学大。归一化让所有特征公平参与训练。

### 第 ③ 步：构造监督学习格式

"模型要知道：用 t-1 时刻的 4 个特征，预测 t 时刻的电力负荷"

原始数据是时间流水账，需要变成"已知条件 → 答案"的格式：

```
输入 X（t-1 时刻的 4 个特征）          输出 Y（t 时刻要预测的值）
var1(t-1) var2(t-1) var3(t-1) var4(t-1)   var1(t)    ← 只预测电力负荷
```

用 `series_to_supervised()` 完成这个转换。**核心概念：把时间序列数据变成一道填空题——给你昨天的值，让你填今天的值。**

### 第 ④ 步：划分训练集 / 测试集

"拿大部分数据训练，留一小部分考试用"

```python
n_train_hours = 365 * 24          # 前 365 天训练
train = values[:n_train_hours, :]
test  = values[n_train_hours:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X,  test_y  = test[:, :-1],  test[:, -1]
```

然后 reshape 成 LSTM 要求的 3D 格式：`[样本数, 时间步, 特征数]`

### 第 ⑤ 步：搭建 + 训练模型

"搭一个简单的 LSTM 网络，让它学 X → Y 的映射关系"

```python
model = Sequential()
model.add(LSTM(64, input_shape=(1, 4)))    # LSTM 层，64 个神经元
model.add(Dense(1))                         # 输出层，1 个数字
model.compile(loss='mae', optimizer='adam')
model.fit(train_X, train_y, epochs=50, ...)
```

`epochs=50` = 把训练数据反复学 50 遍，每遍 loss 都在变小。

### 第 ⑥ 步：预测 + 评估

"用没见过的测试数据考模型，看预测值和真实值差多少"

- **RMSE**：预测值和真实值的平均偏差（数值带单位）
- **MAPE**：误差的百分比（预测偏离了真实值百分之几）

**注意：** 模型输出是归一化后的值（0~1 之间），需要**反归一化**恢复到原始尺度：

```python
inv_yhat = scaler.inverse_transform(...)  # 恢复原始数值（700~900）
```

### 新手写代码的建议

**每一步跑完都 `print()` 看看输出对不对，对了再写下一步。**

| 顺序 | 做什么 | 验证方式 |
|------|--------|---------|
| 1 | `dataset = read_csv(...)` | `print(dataset.head())` 看看读对了没 |
| 2 | 归一化 | `print(scaled[:5])` 看看缩放到 0~1 了没 |
| 3 | 构造 X/Y | `print(reframed.head())` 看看 X 和 y 的对应关系 |
| 4 | 划分训练/测试 | `print(train_X.shape, test_X.shape)` 看看尺寸 |
| 5 | 搭模型 + 训练 | 看 loss 曲线有没有下降 |
| 6 | 预测 + RMSE/MAPE | 看最终数字 |

---

## 六、def 函数 vs 主程序

### 6.1 执行时机

**`def` 写在前面只定义，不执行。**

```
第 1-13  行  import 语句        ← 真执行：加载外挂库
第 17-24 行  def MAPE(...)      ← 不执行！只是"备忘"
第 27-50 行  def series_to...   ← 不执行！同上

第 58 行  dataset = read_csv()  ← 真正开始执行了！
...       一直按顺序执行到第 120 行 ...

期间：
  第 69 行 调用 series_to_supervised() ← 这时才跳到第 27 行执行函数体
  第 119 行 调用 MAPE()                ← 这时才跳到第 17 行执行函数体
```

比喻：
- `def` = 写菜谱贴在墙上（不炒菜）
- `函数名()` = 真正下厨炒菜

Python 永远从上往下读，但 `def` 只定义不执行。函数体只有被 `函数名()` 调用时才执行。

### 6.2 什么时候抽成 def：三问判断法

| 问题 | 回答 YES | 回答 NO |
|------|----------|---------|
| ① 这段代码会用到 2 次以上吗？ | **必须 def** | → 看下一问 |
| ② 这段代码超过 10 行吗？ | **建议 def** | → 看下一问 |
| ③ 给这段代码起个名字，别人一看就懂吗？ | **可以 def** | 留主程序里 |

**总结：**

- 用 2 次以上 → 必须 def
- 用 1 次但超过 10 行且功能独立 → 建议 def
- 用 1 次且只有几行 → 直接写主程序里

`def` 的本质 = 给一段代码起个名字，让主程序读起来像"步骤清单"。func 不是语法要求，是"为了方便"。

---

## 七、开源软件生态

**为什么有人免费写这些库？不全是公益。**

| 库 | 谁养着 |
|---|--------|
| **pandas** | 核心作者被对冲基金雇佣，专职写 pandas |
| **TensorFlow / Keras** | Google 雇工程师全职开发 |
| **numpy** | 多位作者被高校、NASA、Google 等机构雇佣 |
| **scikit-learn** | 法国 INRIA 研究院出钱 |
| **PyTorch** | Meta（Facebook）养着 |

**逻辑：** 公司自己也要用这些工具，与其每家自己写一套，不如让员工写一个最好的，全世界一起用。

个人作者的回报：写开源库 = 最好的简历。pandas 作者出书、演讲、被对冲基金高薪抢着雇、后来创办自己的公司。

"免费"不等于"公益"。这些库免费给你用，但背后是公司/高校在买单，因为他们自己也需要用。

---

## 八、各组件详解

### 1. `series_to_supervised()` — 把"时间流水账"变成"问答对"

原始数据按时间顺序一行行记录，不能直接训练模型。需要变成 **"已知条件 → 答案"** 的格式：

| 已知：1点的负荷/温度/湿度/风速 | 答案：2点的负荷 |
|---|---|

`series_to_supervised()` 做**滑动窗口转换**。本项目用 `n_in=1`（用前 1 个时刻），所以是 t-1 时刻的 4 个特征 → t 时刻的负荷。

### 2. `MinMaxScaler` — 归一化

把所有数据映射到 **[0, 1]** 区间，让每个特征公平参与训练。

### 3. 训练集/测试集划分

- **训练集**：前 365 天的数据（8760 条）→ 模型从这里学习规律
- **测试集**：365 天之后的数据 → 检验模型

### 4. LSTM（64 单元）

LSTM = **Long Short-Term Memory（长短期记忆网络）**。

为什么选它？电力负荷有时序依赖——早上 8 点负荷高，半夜 2 点负荷低。普通神经网络会"忘记"，LSTM 有记忆机制。64 是神经元数量。

### 5. Dense(1) — 输出层

LSTM 提取特征后，Dense(1) 把结果转化为一个单一的预测值——下个时刻的电力负荷。

### 6. Adam 优化器

不断调整网络参数，让预测值越来越接近真实值。Adam 收敛快、能自动调整学习步长。

### 7. Epochs=50

把所有训练数据反复学 50 遍。少了欠拟合，多了过拟合。

### 8. Batch=72 — 每次用多少条样本算一次"怎么改参数"

训练不是一条一条改参数，也不是看完所有数据再改——而是**攒一小撮（batch），算完平均误差，改一次参数**。

#### 8.1 三种极端对比

假设训练集有 **8760** 条样本：

| | batch_size=1 | batch_size=72 | batch_size=8760 |
|:--|:--|:--|:--|
| 名字 | SGD（随机梯度下降） | Mini-batch（小批量） | Batch GD（批量梯度下降） |
| 每轮改几次参数 | 8760 次 | 8760÷72 ≈ 122 次 | 1 次 |
| 训练速度 | 🐢 极慢 | 🚀 快 | 🐢 慢（一次算全部很重） |
| 内存占用 | 很小 | 适中 | 极大（可能爆显存） |
| 收敛稳定性 | 😵 震荡大（每条都改，方向跳来跳去） | ✅ 比较稳 | 📉 太稳，容易卡在局部最优 |

#### 8.2 图解

```
batch_size = 1（一条一条改）：
  样本1 → 改参数 → 样本2 → 改参数 → 样本3 → 改参数 → ...
  缺点：每一步方向随机，像醉汉走路，晃得厉害

batch_size = 72（一次攒 72 条再改）：
  [样本1~72] → 算平均梯度 → 改一次参数
  [样本73~144] → 算平均梯度 → 改一次参数
  ✅ 方向稳定，计算高效

batch_size = 8760（全看完再改）：
  [全部 8760 条] → 算平均梯度 → 改一次参数
  缺点：一次算全部太重，且梯度太"平滑"，容易陷入局部最优
```

#### 8.3 为什么这里是 72？

72 小时 = **3 天**的电力数据。

电力负荷有明显的日周期（24小时），72 = 24 × 3，刚好覆盖 3 个完整日周期，既能抓到一天的规律，又不会太大。

| batch_size | 含义 | 效果 |
|:--|:--|:--|
| 24 | 1 天 | 可能只抓到单日模式 |
| **72** | **3 天** | ✅ 均衡：有统计意义，又不过大 |
| 168 | 7 天 | 可以，但每轮更新次数变少 |

#### 8.4 影响什么？

| 影响 | 说明 |
|:--|:--|
| **训练速度** | batch 太小 → 改参数次数多 → 慢；batch 太大 → 一次计算重 → 也慢 |
| **收敛稳定性** | batch 太小 → 梯度方向乱跳；batch 太大 → 梯度太"平均" |
| **内存/显存** | batch 越大越吃内存 |
| **泛化能力** | 小 batch 有一定噪声，反而有助于跳出局部最优（正则化效果） |

#### 8.5 怎么选？经验值

| 数据量 | 推荐 batch_size |
|:--|:--|
| < 1000 条 | 32 ~ 128 |
| 1000 ~ 10000 条 | 64 ~ 256 |
| > 10000 条 | 128 ~ 512 |

**时间序列额外考虑：** 取周期的整数倍（如 24、48、72、168），让每个 batch 包含完整周期。

> **一句话：batch_size 决定了"攒多少条样本的误差平均一下再改参数"。太小震荡大，太大收敛慢，72（3天）是这里的一个经验好选择。**

### 9. RMSE 和 MAPE — 考试评分

| 指标 | 全称 | 含义 |
|------|------|------|
| **RMSE** | 均方根误差 | 预测值和真实值的平均偏差（带单位） |
| **MAPE** | 平均绝对百分比误差 | 预测误差占真实值的百分比 |

---

## 九、LSTM 底层原理

### 9.1 普通神经网络为什么"没记性"

```python
# Dense 神经元：t=1 时刻
h1 = activation(W · x1 + b)   # 算完，h1 被丢弃

# Dense 神经元：t=2 时刻
h2 = activation(W · x2 + b)   # 全新计算，对 t=1 一无所知
```

### 9.2 LSTM 的核心创新：细胞状态 C（信息传送带）

```python
# LSTM 神经元：t=1 时刻
h1, C1 = lstm_step(x1, h0, C0)   # C1 被保留下来！

# LSTM 神经元：t=2 时刻
h2, C2 = lstm_step(x2, h1, C1)   # C1 传进来了！
```

**记忆的本质：一个变量 C 从 t=1 → t=2 → t=3...永不丢弃。**

### 9.3 三个门控制记忆

| 门 | 作用 | 类比 |
|----|------|------|
| **遗忘门** | 决定丢掉 C 中哪些旧信息 | 撕掉日记本里不重要的旧页 |
| **输入门** | 决定把哪些新信息写入 C | 在日记本里记下新发现 |
| **输出门** | 决定从 C 中提取什么输出 | 根据日记回答当前提问 |

每个门本质上是一个带 **sigmoid 激活函数**的小型计算（sigmoid 输出 0~1，0=完全关闭，1=完全打开）。

### 9.4 三个门完全自动学习，不需要手动设置

```python
model.add(LSTM(64, ...))   # 你只写了这一行
```

Keras 自动为 64 个神经元创建并训练：每个神经元的遗忘门参数（64 套）、输入门参数（64 套）、输出门参数（64 套）、细胞状态 C（64 条）。

训练过程中，**Adam 优化器自动调整所有这些门参数**，完全不用手动干预。

### 9.5 64 可以改吗？怎么选？

```python
model.add(LSTM(64, ...))   # 当前
model.add(LSTM(128, ...))  # 可以改成 128
model.add(LSTM(32, ...))   # 可以改成 32
```

| 神经元数 | 效果 | 适用场景 |
|---------|------|---------|
| **太少**（8~16） | 模型太简单，欠拟合 | 数据量很小 |
| **适中**（32~128） | ✅ 通常最好 | 大多数中小规模数据 |
| **太多**（256+） | 可能过拟合，训练慢 | 海量数据 |

**选法：** 尝试 32/64/128，对比哪个 RMSE/MAPE 最低。

### 9.6 LSTM 层可设置的常用参数

| 参数 | 含义 | 默认值 | 本项目值 |
|------|------|--------|---------|
| `units` | 神经元数量（必须设置） | 无 | 64 |
| `return_sequences` | 是否输出每个时间步 | `False` | 默认 |
| `dropout` | 输入随机丢弃比例 | 0 | 默认 |
| `recurrent_dropout` | 循环连接随机丢弃比例 | 0 | 默认 |
| `input_shape` | 输入形状 | 无 | (1, 4) |

### 9.7 框架底层也是 Python 写的

Keras 的 LSTM 源码在 `keras/src/layers/rnn/lstm.py`，简化后的内部逻辑：

```python
class LSTM:
    def step(self, x_t, h_prev, C_prev):
        f_t = sigmoid(x_t @ W_f + h_prev @ U_f + b_f)  # ①遗忘门
        i_t = sigmoid(x_t @ W_i + h_prev @ U_i + b_i)  # ②输入门
        C_tilde = tanh(x_t @ W_c + h_prev @ U_c + b_c) # 候选记忆
        C_new = f_t * C_prev + i_t * C_tilde           # ③更新C
        o_t = sigmoid(x_t @ W_o + h_prev @ U_o + b_o)  # ④输出门
        h_new = o_t * tanh(C_new)
        return h_new, C_new
```

### 9.8 深度学习编程 ≠ 手写神经网络

`lstm_predict.py` 120 行代码的真实分工：

- ① 数据读取与清洗（~10行）
- ② 特征工程（~20行）← 最需要人脑判断
- ③ 数据归一化与反向还原（~15行）
- ④ 数据集划分与 reshape（~10行）
- ⑤ 模型定义（~5行）← 只有这部分是"设参数"
- ⑥ 训练与评估（~30行）
- ⑦ 可视化（~10行）

**框架帮你省掉的：** LSTM 内部的门控计算、梯度反向传播。  
**你必须自己写的：** 数据怎么进来、怎么出去、怎么评估。

### 9.9 什么时候需要 LSTM？

**判断标准：当前输出是否依赖历史输入？**

✅ 需要 LSTM（有时序依赖）：电力负荷预测、股票预测、机器翻译、视频分析、语音识别  
❌ 不需要 LSTM（无时序依赖）：图片分类、房价预测、垃圾邮件分类、疾病诊断

### 9.10 一句话总结

```
记忆的本质 = 一个名为 C 的向量在时间步之间传递
三个门的作用 = 三个 sigmoid 函数，分别控制 C 的保留/写入/输出
什么时候用 LSTM = 数据有时序依赖时 ✅ / 独立数据时 ❌
长短期 = 既能记住遥远过去（C 不衰减传递）+ 也能处理眼前新信息