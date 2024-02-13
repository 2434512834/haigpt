"""

准备莎士比亚数据集进行字符级语言建模。
所以我们不使用 GPT-2 BPE 令牌进行编码，而是直接将字符映射为整数。
将保存包含 id 的 train.bin 和 val.bin，以及包含编码器、解码器和其他相关信息的 meta.pkl。

"""




# 导入所需的库
import os
import pickle
import requests
import numpy as np

# 下载 tiny shakespeare 数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'train.txt') # 当前脚本文件下的'train.txt'
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # 如果数据集不存在，则从网上下载
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# 读取数据集
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 获取文本中所有的唯一字符
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# 创建字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # 编码器：输入字符串，输出整数列表
def decode(l):
    return ''.join([itos[i] for i in l]) # 解码器：输入整数列表，输出字符串

# 创建训练和测试数据集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 将训练和测试数据集编码为整数
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 将训练和测试数据集导出为 bin 文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 保存元信息，以便我们以后进行编码/解码
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
