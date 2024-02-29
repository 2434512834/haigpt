from transformers import GPT2Config

def get_config():
    config = GPT2Config(
        vocab_size=50257,  # 词汇表大小  64000
        hidden_size=16,  # 隐藏层大小 # 768
        num_layers=8,  # 层数
        num_attention_heads=16,  # 注意力头数
        intermediate_size=16  # 中间层大小
        max_position_embeddings=768,  # 最大位置嵌入
        layer_norm_epsilon=1e-5,  # 层归一化 epsilon
        initializer_range=0.02,  # 初始化范围
        dropout=0.1,  # dropout 比例
        attention_dropout=0.1,  # 注意力 dropout 比例
        bos_token_id=0,  # 起始 token id
        eos_token_id=1,  # 结束 token id
        pad_token_id=50256,  # padding token id
        gradient_checkpointing=False,  # 是否使用梯度检查点
        use_cache=True,  # 是否使用缓存
        bos_token="",  # 起始 token
        eos_token="",  # 结束 token
    )
    return config
