import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from models.configuration_baichuan import BaiChuanConfig
from models.modeling_baichuan import BaiChuanForCausalLM
from torch.cuda.amp import GradScaler, autocast
import logging
import time
from prepare_data import load_preprocessed_data
from tqdm import tqdm

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache_path", type=str, default="data_dir/train_data.pt", help="预处理后的训练数据路径")
    parser.add_argument("--val_cache_path", type=str, default="data_dir/val_data.pt", help="预处理后的验证数据路径")
    parser.add_argument("--steps_per_epoch", type=int, default=1, help="每个 epoch 的步骤数")
    parser.add_argument("--checkpoint_saving_path", type=str, default="checkpoints", help="保存检查点文件的路径")
    parser.add_argument("--resume", type=str, choices=['scratch', 'resume', 'other'], default='resume', help="初始化选项：'scratch'表示新模型，'resume'表示恢复训练，'other'表示其他选项")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--dtype", type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help="数据类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="每隔多少个 epoch 保存一次检查点")
    parser.add_argument("--num_workers", type=int, default=4, help="加载数据时使用的线程数")
    parser.add_argument("--use_multiprocessing", action="store_true", help="是否使用多进程加载数据")
    return parser

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger()

arg_parser = get_argument_parser()
args = arg_parser.parse_args()

log_path = "training.log"
logger = setup_logging(log_path)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import re

def prepare_model():
    config = BaiChuanConfig()
    model = BaiChuanForCausalLM(config).to(torch.device("cuda"))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0

    if args.resume == 'scratch':
        logger.info("从头开始初始化一个新模型")
        config.save_pretrained(args.checkpoint_saving_path)
    elif args.resume == 'resume':
        logger.info(f"尝试从检查点恢复训练")
        checkpoint_files = [f for f in os.listdir(args.checkpoint_saving_path) if f.startswith("checkpoint_epoch") and f.endswith(".pt")]
        if checkpoint_files:
            epoch_numbers = [int(re.search(r'\d+', f).group()) for f in checkpoint_files if re.search(r'\d+', f)]
            if not epoch_numbers:
                raise ValueError("检查点文件名中无法找到 epoch 数字")
            max_epoch = max(epoch_numbers)
            latest_checkpoint_file = f"checkpoint_epoch{max_epoch}.pt"
            checkpoint_path = os.path.join(args.checkpoint_saving_path, latest_checkpoint_file)
            logger.info(f"找到最新的检查点文件：{latest_checkpoint_file}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            raise FileNotFoundError("未找到任何检查点文件")
    else:
        raise ValueError("无效的初始化选项。使用 'scratch'、'resume' 或 'other'。")

    return model, optimizer, start_epoch

def evaluate(data_loader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="评估进度", leave=False):
            data = data.cuda(non_blocking=True)
            loss = model(data, labels=data).loss
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train(data_loader, model, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    scaler = GradScaler(enabled=args.dtype in ['float16', 'bfloat16'])
    start_time = time.time()
    for step, data in enumerate(tqdm(data_loader, desc="训练进度", leave=False)):
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
    average_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch + 1}, 平均损失: {average_loss}")
    print(f"Epoch {epoch + 1}, 平均损失: {average_loss}")

if __name__ == "__main__":
    train_data, val_data = load_preprocessed_data(args.train_cache_path, args.val_cache_path)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=args.num_workers, multiprocessing_context="spawn" if args.use_multiprocessing else None)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=args.num_workers, multiprocessing_context="spawn" if args.use_multiprocessing else None)

    model, optimizer, start_epoch = prepare_model()
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=args.epochs * len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"开始训练第 {epoch + 1} 轮")
        train(train_loader, model, optimizer, scheduler, epoch)
        if (epoch + 1) % args.checkpoint_interval == 0:
            val_loss = evaluate(val_loader, model)
            logger.info(f"第 {epoch + 1} 轮结束，验证损失: {val_loss}")
            print(f"第 {epoch + 1} 轮结束，验证损失: {val_loss}")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1},
                       os.path.join(args.checkpoint_saving_path, f"checkpoint_epoch{epoch + 1}.pt"))
        start_epoch = epoch + 1
