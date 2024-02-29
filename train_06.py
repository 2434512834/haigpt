import os
import argparse
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from models.configuration_baichuan import BaiChuanConfig
from models.modeling_baichuan import BaiChuanForCausalLM
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm
import time

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_dir", help="预训练文本文件目录")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="分词器模型文件路径")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="每个GPU的微批处理大小")
    parser.add_argument("--max_length", type=int, default=4096, help="语料中每个句子的最大标记数")
    parser.add_argument("--steps_per_epoch", type=int, default=1, help="每个 epoch 的步骤数")
    parser.add_argument("--checkpoint_saving_path", type=str, default="checkpoints", help="保存检查点文件的路径")
    parser.add_argument("--init_from", type=str, choices=['scratch', 'resume', 'other'], default='scratch', help="初始化选项：'scratch'表示新模型，'resume'表示恢复训练，'other'表示其他选项")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--dtype", type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help="数据类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="每隔多少个 epoch 保存一次检查点")
    parser.add_argument("--num_workers", type=int, default=4, help="加载数据时使用的线程数")
    return parser

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger()

arg_parser = get_argument_parser()
args = arg_parser.parse_args()

log_path = "training.log"
logger = setup_logging(log_path)

class DataEngine(Dataset):
    def __init__(self, data_dir, tokenizer_path, micro_batch_size, max_length):
        self.MIN_TEXT_LEN = 20
        self.EOS_TOKEN_ID = 2
        self.data_dir = data_dir
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.data = []
        self.global_input_paths = [self.data_dir + "/" + x for x in os.listdir(self.data_dir)]
        self.load_data()

    def load_data(self):
        all_data = []
        for file_path in self.global_input_paths:
            data = []
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_id, line in enumerate(f):
                    cc = self.sp.EncodeAsIds(line.strip()) + [self.EOS_TOKEN_ID]
                    if len(cc) < self.MIN_TEXT_LEN:
                        cc = []
                    data.extend(cc)
                    if len(data) >= self.micro_batch_size * (self.max_length + 1):
                        index = self.micro_batch_size * (self.max_length + 1)
                        all_data.append(data[:index])
                        data = []
        # 将所有数据转换为张量并存储在内存中
        self.data = [torch.LongTensor(np.asarray(seq).reshape(self.micro_batch_size, self.max_length + 1)) for seq in all_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_model():
    config = BaiChuanConfig()
    model = BaiChuanForCausalLM(config).to(torch.device("cuda"))  # 将模型移到GPU上
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if args.init_from == 'scratch':
        logger.info("从头开始初始化一个新模型")
        # 保存模型配置文件
        config.save_pretrained(args.checkpoint_saving_path)
    elif args.init_from == 'resume':
        logger.info(f"从检查点 {args.checkpoint_saving_path} 恢复训练")
        checkpoint_path = os.path.join(args.checkpoint_saving_path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise FileNotFoundError(f"未找到检查点文件：{checkpoint_path}")
    else:
        raise ValueError("无效的初始化选项。使用'scratch'、'resume'或'other'。")
    return model, optimizer

def evaluate(data_loader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            data = data.cuda(non_blocking=True)
            loss = model(data, labels=data).loss
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train(data_loader, model, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    scaler = GradScaler(enabled=args.dtype in ['float16', 'bfloat16'])
    start_time = time.time()
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{args.epochs}") as progress_bar:
        for step, data in enumerate(data_loader):
            data = data.cuda(non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=args.dtype in ['float16', 'bfloat16']):
                loss = model(data, labels=data).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % args.steps_per_epoch == 0:
                logger.info(f"步骤 {step}, 损失: {loss.item()}")
            progress_bar.update(1)
            if step % 10 == 0:  # Log every 10 steps
                elapsed_time = time.time() - start_time
                log_message = f"迭代次数: {step}, 损失: {loss.item():.4f}, 耗时: {elapsed_time*1000:.2f}ms"
                logger.info(log_message)
                print(log_message)  # Print the same log message to the console
    average_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch + 1}, 平均损失: {average_loss}")
    print(f"Epoch {epoch + 1}, 平均损失: {average_loss}")

if __name__ == "__main__":
    data_engine = DataEngine(args.data_dir, args.tokenizer_path, args.train_micro_batch_size_per_gpu, args.max_length)
    train_loader = DataLoader(data_engine, batch_size=None, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    model, optimizer = prepare_model()
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=args.epochs * len(train_loader))

    for epoch in range(args.epochs):
        logger.info(f"开始训练第 {epoch + 1} 轮")
        train(train_loader, model, optimizer, scheduler, epoch)
        if (epoch + 1) % args.checkpoint_interval == 0:
            val_loss = evaluate(train_loader, model)
            logger.info(f"第 {epoch + 1} 轮结束，验证损失: {val_loss}")
            print(f"第 {epoch + 1} 轮结束，验证损失: {val_loss}")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(args.checkpoint_saving_path, f"checkpoint_epoch{epoch + 1}.pt"))  # 保存检查点
