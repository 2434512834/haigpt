"""
GPT 语言模型的完整定义，全部都在这个单个文件中。

参考资料：
1）OpenAI发布的官方GPT-2 TensorFlow实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2）huggingface/transformers PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

"""

# 导入所需的库
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# 定义 LayerNorm 类，继承自 PyTorch 的 nn.Module
class LayerNorm(nn.Module):
    """ 
    LayerNorm 类中包含一个可选的偏置项。
    PyTorch 的 LayerNorm 不支持直接设置 bias=False，所以我们自定义了一个 LayerNorm 类，它允许在初始化时通过 bias 参数来选择是否包含偏置项。
    """
    # 初始化函数
    def __init__(self, ndim, bias):
        # 调用父类的初始化函数
        super().__init__()
        # 定义权重参数，初始化为全 1 向量
        self.weight = nn.Parameter(torch.ones(ndim))
        # 如果 bias 为 True，则定义偏置参数，初始化为全 0 向量；否则偏置为 None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    # 前向传播函数
    def forward(self, input):
        # 对输入进行层归一化处理，如果偏置不为 None，则包含偏置项
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力模块，它是 Transformer 模型中的关键部分。
    这个模块的主要功能是计算输入序列中每个元素的注意力分数，
    并根据这些分数来更新元素的表示。
    """

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度可以被头的数量整除
        assert config.n_embd % config.n_head == 0
        # 所有头的键、查询、值的三种投影，但是在一个批次（batch）中所有数据送入模型并行处理。
        # ，"投影"（projection）通常指的是将数据从一个维度空间转换到另一个维度空间的过程。这个过程通常通过矩阵乘法来实现。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 头的数量
        self.n_head = config.n_head
        # 嵌入维度
        self.n_embd = config.n_embd
        # dropout率
        self.dropout = config.dropout
        # flash attention 是优化注意力计算方法，更有效利用GPU加速，但只在PyTorch >= 2.0中支持

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告: 使用慢速注意力。Flash Attention需要PyTorch >= 2.0")
            # 因果掩码，确保注意力只应用于输入序列的左侧（或之前）的信息，不能够看到右侧（或之后）的信息。因此，我们会使用一个掩码来阻止模型 "看到" 未来的信息。
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B: 批次大小(batch size)，T: 序列长度(sequence length)，C: 嵌入维度(embedding dimensionality (n_embd))  
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算批次中所有头的查询(query)、键(key)、值(values)，并将头向前移动成为批次维度
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力; 自我关注: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 使用Flash Attention CUDA内核进行高效注意力
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手动实现注意力
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 并排重新组装所有头输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # 定义一个线性层，输入维度为config.n_embd，输出维度为4 * config.n_embd，是否添加偏置由config.bias决定
        self.gelu    = nn.GELU() # 定义一个GELU激活函数
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # 定义一个线性层，输入维度为4 * config.n_embd，输出维度为config.n_embd，是否添加偏置由config.bias决定
        self.dropout = nn.Dropout(config.dropout) # 定义一个Dropout层，丢弃概率为config.dropout

    def forward(self, x):
        x = self.c_fc(x) # 通过线性层处理输入x
        x = self.gelu(x) # 通过GELU激活函数处理x
        x = self.c_proj(x) # 通过线性层处理x
        x = self.dropout(x) # 通过Dropout层处理x
        return x # 返回处理后的x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) # 定义一个层归一化（LayerNorm）层，输入维度为config.n_embd，是否添加偏置由config.bias决定
        self.attn = CausalSelfAttention(config) # 定义一个因果自注意力（CausalSelfAttention）层，配置参数为config
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias) # 定义一个层归一化（LayerNorm）层，输入维度为config.n_embd，是否添加偏置由config.bias决定
        self.mlp = MLP(config) # 定义一个多层感知机（MLP）层，配置参数为config

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # 先通过第一个层归一化层处理输入x，然后通过因果自注意力层处理，最后与原始的x相加
        x = x + self.mlp(self.ln_2(x)) # 先通过第二个层归一化层处理输入x，然后通过多层感知机层处理，最后与原始的x相加
        return x # 返回处理后的x

@dataclass
class GPTConfig:
    block_size: int = 1024 # 块大小，通常对应于模型可以一次处理的最大序列长度
    vocab_size: int = 50304 # 词汇表大小，GPT-2的词汇表大小为50257，为了效率，填充到最近的64的倍数
    n_layer: int = 12 # Transformer模型的层数
    n_head: int = 12 # 自注意力机制的头的数量
    n_embd: int = 768 # 嵌入维度，也是每个头的大小
    dropout: float = 0.0 # Dropout层的丢弃概率
    bias: bool = True # 是否在线性层和层归一化（LayerNorm）层中添加偏置，True表示像GPT-2一样添加偏置，False表示不添加偏置，可能会更好更快

# ----------
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None # 确保词汇表大小不为空
        assert config.block_size is not None # 确保块大小不为空
        self.config = config # 保存配置参数

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # 定义词嵌入层，输入维度为词汇表大小，输出维度为嵌入维度
            wpe = nn.Embedding(config.block_size, config.n_embd), # 定义位置嵌入层，输入维度为块大小，输出维度为嵌入维度
            drop = nn.Dropout(config.dropout), # 定义Dropout层，丢弃概率为config.dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 定义Transformer的层数，每一层都是一个Block
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # 定义层归一化（LayerNorm）层，输入维度为嵌入维度，是否添加偏置由config.bias决定
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 定义线性层，输入维度为嵌入维度，输出维度为词汇表大小，不添加偏置
        # 当使用 torch.compile() 进行权重绑定会生成一些警告：
        # “UserWarning: functional_call 传递了多个绑定权重值。
        # 此行为已被弃用，并且在未来版本中将成为错误”
        # 对此还不100% 完全确定，到目前为止似乎无害。TODO 需要进一步调查
        self.transformer.wte.weight = self.lm_head.weight # 将词嵌入层的权重和线性层的权重绑定在一起，实现权重共享 # https://paperswithcode.com/method/weight-tying

        # 初始化所有权重
        self.apply(self._init_weights)
        # 根据GPT-2论文，将特殊缩放的初始化应用于残差投影。
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。
        对于非嵌入计数（默认），位置嵌入会被减去。
        词嵌入也会被减去，但由于参数共享，这些参数实际上被用作最后一层的权重，所以我们将它们包含在内。
        """
        n_params = sum(p.numel() for p in self.parameters()) # 计算所有参数的数量
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() # 如果非嵌入计数，减去位置嵌入的参数数量
        return n_params # 返回参数数量

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 如果模块是线性层，对权重进行正态分布初始化
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # 如果存在偏置，将偏置初始化为0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 如果模块是嵌入层，对权重进行正态分布初始化
            

    def forward(self, idx, targets=None):
        device = idx.device # 获取设备信息
        b, t = idx.size() # 获取输入的批次大小和序列长度
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}" # 确保序列长度不超过块大小
        pos = torch.arange(0, t, dtype=torch.long, device=device) # # 创建位置信息 # shape (t)就是说这个位置信息数组的形状是(t,)。

        # 前向传播GPT模型本身
        tok_emb = self.transformer.wte(idx) # # 获取形状为 (b, t, n_embd) 的词嵌入
        pos_emb = self.transformer.wpe(pos) # 获取形状 (t, n_embd) 的位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb) # 对词嵌入和位置嵌入求和并进行dropout
        for block in self.transformer.h:
            x = block(x) # 通过每一层Transformer
        x = self.transformer.ln_f(x) # 进行层归一化

        if targets is not None:
            # 如果给我们一些期望的目标，也计算损失函数
            logits = self.lm_head(x) # 计算logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # 计算交叉熵损失
        else:
            # 推理时间小型优化：仅在最后一个位置转发 lm_head
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # # 在推理时，只计算最后一个位置的logits
            loss = None # 没有目标，所以损失为None

        return logits, loss # 返回logits和损失

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size # 确保新的块大小不超过当前的块大小
        self.config.block_size = block_size # 更新块大小
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) # 调整位置嵌入的权重大小
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size] # 调整注意力层的偏置大小

        #          以下的注释

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # 确保模型类型在预定义的模型类型中
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # 如果没有提供override_args，则默认为空字典
        override_args = override_args or {} # 默认为空字典
        # 只有dropout可以被覆盖，参见下面的注释
        assert all(k == 'dropout' for k in override_args)
        # 导入transformers库中的GPT2LMHeadModel
        from transformers import GPT2LMHeadModel
        # 打印正在从预训练的gpt模型中加载权重的信息
        print("loading weights from pretrained gpt: %s" % model_type)

        # 根据模型类型确定n_layer, n_head 和 n_embd的值
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M 参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M 参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 参数
        }[model_type]
        # 强制设置vocab_size=50257, block_size=1024, bias=True
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # GPT模型检查点的vocab_size总是50257
        config_args['block_size'] = 1024 # GPT模型检查点的block_size总是1024
        config_args['bias'] = True # GPT模型检查点的bias总是True
        # 如果需要，我们可以覆盖dropout率
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 创建一个从零开始初始化的minGPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        # 获取模型的状态字典
        sd = model.state_dict()
        # 获取状态字典的键
        sd_keys = sd.keys()
        # 丢弃这个mask / buffer，它不是一个参数
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 初始化一个huggingface/transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # 获取huggingface/transformers模型的状态字典
        sd_hf = model_hf.state_dict()


        # 确保所有参数都对齐并且在名称和形状上匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是一个缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同样，只是一个掩码（缓冲区）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上openai的检查点使用了"Conv1D"模块，但我们只想使用普通的线性模块
        # 这意味着我们在导入它们时必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对我们需要转置的Conv1D权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 对其他参数进行普通的复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何2D的参数都会被衰减，否则不会。
        # 即，所有在矩阵乘法和嵌入中的权重张量都会衰减，所有的偏置和层归一化都不会。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"衰减的参数张量数量: {len(decay_params)}, 参数数量: {num_decay_params:,}")
        print(f"未衰减的参数张量数量: {len(nodecay_params)}, 参数数量: {num_nodecay_params:,}")
        # 创建AdamW优化器，如果可用的话使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用融合的AdamW: {use_fused}")

        return optimizer



    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤出不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何2D的参数都会进行权重衰减，否则不会。
        # 即，所有在矩阵乘法和嵌入中的权重张量都会衰减，所有的偏置和层归一化不会。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},  # 需要衰减的参数组
            {'params': nodecay_params, 'weight_decay': 0.0}  # 不需要衰减的参数组
        ]
        # 计算需要衰减和不需要衰减的参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 创建AdamW优化器，如果可用的话，使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer  # 返回配置好的优化器

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估计模型的浮点运算利用率（MFU），单位为A100 bfloat16峰值FLOPS """
        # 首先估计我们每次迭代的浮点运算次数。
        # 参考PaLM论文附录B：https://arxiv.org/abs/2204.02311
        N = self.get_num_params()  # 获取模型参数数量
        cfg = self.config  # 获取模型配置
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size  # 获取层级、头部、嵌入维度和块大小
        flops_per_token = 6*N + 12*L*H*Q*T  # 计算每个token的浮点运算次数
        flops_per_fwdbwd = flops_per_token * T  # 计算每次前向和后向传播的浮点运算次数
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # 计算每次迭代的浮点运算次数
        # 将我们的浮点运算吞吐量表示为A100 bfloat16峰值浮点运算次数的比例
        flops_achieved = flops_per_iter * (1.0/dt)  # 每秒实现的浮点运算次数
        flops_promised = 312e12  # A100 GPU的bfloat16峰值浮点运算次数是312 TFLOPS
        mfu = flops_achieved / flops_promised  # 计算模型的浮点运算利用率
        return mfu  # 返回模型的浮点运算利用率

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        接收一个索引序列idx（LongTensor，形状为(b,t)），并完成序列max_new_tokens次，
        每次都将预测结果反馈给模型。大多数情况下，你可能需要确保模型处于model.eval()模式。
        """
        for _ in range(max_new_tokens):  # 对于每一个新的token
            # 如果序列上下文过长，我们必须将其裁剪至block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播模型以获取序列中索引的logits
            logits, _ = self(idx_cond)
            # 取出最后一步的logits并按照期望的温度进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选地将logits裁剪至仅包含前k个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用softmax将logits转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样的索引添加到运行的序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx  # 返回生成的序列
