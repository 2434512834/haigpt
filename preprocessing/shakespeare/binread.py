"""
写一个读取当前代码文件下的train.bin文件的python代码，并且打印前10行



"""
import numpy as np

# 读取 train.bin 文件
data = np.fromfile('train.bin', dtype=np.uint16)

# 打印读取到的前10行数据
print(data[:100])