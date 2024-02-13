"""
这段代码的主要功能是下载 tiny shakespeare 数据集，将数据集分为训练集和验证集，然后使用 tiktoken 的 gpt2 bpe 对数据进行编码，并将编码后的数据保存为 bin 文件。
这段代码主要完成了以下几个任务：

1. 下载 tiny shakespeare 数据集：如果当前目录下不存在数据集文件，就从网上下载并保存到本地。

2. 读取数据集文件：打开数据集文件，读取所有内容，并计算数据的长度。

3. 将数据集分为训练集和验证集：取数据的前90%作为训练集，后10%作为验证集。

4. 使用 tiktoken 的 gpt2 bpe 对数据进行编码：获取 gpt2 的编码器，然后对训练集和验证集的数据进行编码。

5. 打印训练集和验证集的 token 数量：计算训练集和验证集的 token 数量，并打印出来。

6. 将编码后的数据转换为 numpy 数组，并保存为 bin 文件：将训练集和验证集的 token 列表转换为 numpy 数组，然后保存为 bin 文件。

这段代码的主要目的是为了准备数据，以便进行后续的机器学习或深度学习训练。



pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tiktoken



"""

import os  # 导入os模块，用于处理文件和目录
import requests  # 导入requests模块，用于发送HTTP请求
import tiktoken  # 导入tiktoken模块，用于编码和解码文本
import numpy as np  # 导入numpy模块，用于处理大型多维数组和矩阵的数学运算

# 下载 tiny shakespeare 数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')  # 定义数据集文件的路径
if not os.path.exists(input_file_path):  # 检查数据集文件是否存在
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'  # 数据集的下载链接
    # 如果数据集文件不存在，就从网上下载
    with open(input_file_path, 'w', encoding='utf-8') as f:  # 打开文件，准备写入
        f.write(requests.get(data_url).text)  # 从网上下载数据，并写入文件

# 读取数据集文件
with open(input_file_path, 'r', encoding='utf-8') as f:  # 打开文件，准备读取
    data = f.read()  # 读取文件内容
n = len(data)  # 计算数据的长度
# 将数据集分为训练集和验证集
train_data = data[:int(n*0.9)]  # 取前90%的数据作为训练集
val_data = data[int(n*0.9):]  # 取后10%的数据作为验证集

# 使用 tiktoken 的 gpt2 bpe 对数据进行编码
enc = tiktoken.get_encoding("gpt2")  # 获取gpt2的编码器
train_ids = enc.encode_ordinary(train_data)  # 对训练集数据进行编码

val_ids = enc.encode_ordinary(val_data)  # 对验证集数据进行编码

print(f"train has {len(train_ids):,} tokens")  # 打印训练集的token数量
print(f"val has {len(val_ids):,} tokens")  # 打印验证集的token数量

# 将编码后的数据转换为 numpy 数组，并保存为 bin 文件
train_ids = np.array(train_ids, dtype=np.uint16)  # 将训练集的token列表转换为numpy数组
val_ids = np.array(val_ids, dtype=np.uint16)  # 将验证集的token列表转换为numpy数组
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))  # 将训练集的numpy数组保存为bin文件
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))  # 将验证集的numpy数组保存为bin文件

# train.bin 文件包含 301,966 个 tokens
# val.bin 文件包含 36,059 个 tokens