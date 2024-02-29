from transformers import AutoTokenizer
from models.modeling_baichuan import BaiChuanForCausalLM

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/shidonghai/haigpt/train/train_02/converted_model", trust_remote_code=True)

# 加载模型
model = BaiChuanForCausalLM.from_pretrained("/home/shidonghai/haigpt/train/train_02/converted_model")

# 将模型和输入移动到GPU
model = model.to('cuda:0')

# 准备输入文本
inputs = tokenizer('你好\n你好吗？->', return_tensors='pt')
inputs = inputs.to('cuda:0')

# 生成预测
pred = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.1)

# 解码并打印预测结果
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
