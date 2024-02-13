import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设我们有一个词汇表
vocab = ["给我", "一个", "torch", "的", "linux", "安装", "命令"]

# 创建一个词嵌入层
embedding = nn.Embedding(len(vocab), 10)  # 假设我们的嵌入维度是10

# 将我们的文本转换为数值形式
text = torch.tensor([vocab.index(word) for word in vocab])

# 假设我们有一个查询向量
query = torch.rand(1, 10)


def calculate_attention_scores(query, keys, values):
    # 计算查询和键之间的点积
    dot_products = torch.matmul(query, keys.t())

    # 通过softmax函数将点积转换为注意力分数
    attention_scores = F.softmax(dot_products, dim=-1)

    # 使用注意力分数对值进行加权求和，得到输出
    output = torch.matmul(attention_scores, values)

    return attention_scores, output


# 将文本转换为向量
keys = embedding(text)
values = keys  # 在这个例子中，我们假设键和值是相同的

attention_scores, output = calculate_attention_scores(query, keys, values)

print("注意力分数：", attention_scores)
print("输出：", output)
