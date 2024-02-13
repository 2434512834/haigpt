
import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys
import os
import torch
import transformers
from torch.utils.data import DataLoader
from os.path import join

from sklearn.model_selection import train_test_split

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np

# 导入dataset.py
from dataset import MyDataset
# 导入pytorchtools.py
from pytorchtools import EarlyStopping
# 导入data_parallel.py
from data_parallel import BalancedDataParallel
# config.py

# 由于是传入 参数的原因，无法识别，
# from config.config import *
import sys
sys.path.append('/hy-tmp/haigpt')  # 替换为你的实际路径
from config import *
from model.model_01.model import GPTConfig, GPT

# 用于测试 from model.model_01.model import GPTConfig, GPT 导入是否成功
# # 尝试创建一个GPTConfig的实例
# config = GPTConfig()
# # 尝试创建一个GPT的实例
# model = GPT(config)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--cuda', default=True, type=bool, help='是否使用CUDA进行训练')
    # parser.add_argument('--no_cuda', action='store_flase', help='不使用GPU进行训练')
    # parser.add_argument('--cuda', action='store_true', help='使用GPU进行训练')
    parser.add_argument('--vocab_path', default='../../vocab/vocab.txt', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--model_config', default='../../config/config.json', type=str, required=False,
                        help='设置模型参数')
    parser.add_argument('--train_path', default='../../data/train.pkl', type=str, required=False, help='训练集路径')
    parser.add_argument('--max_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--log_path', default='../../log/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--log', default=True, help="是否记录日志")
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    # parser.add_argument('--input_len', default=200, type=int, required=False, help='输入的长度')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--save_interval', type=int, default=1, help='在训练过程中，每隔多少个训练周期（epoch）就保存一次模型。')
    parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help='Precision for training')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=10, type=int, required=False, help='0号卡的batch size')
    parser.add_argument('--lr', default=2.6e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--save_model_path', default='../../model_out', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    # parser.add_argument('--label_smoothing', default=True, action='store_true', help='是否进行标签平滑')
    parser.add_argument('--val_num', type=int, default=8000, help='验证集大小')
    args = parser.parse_args()

    # 从config模块导入的变量覆盖argparse的默认参数
    for arg in vars(args):
        if arg in globals():
            setattr(args, arg, globals()[arg])

    return args

# 通过print查看config.py是否能够覆盖默认参数
args = set_args()
print(args.vocab_path)



def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def load_dataset(logger, args):
    """
    加载训练集和验证集
    """
    logger.info("loading training dataset and validating dataset")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分训练集与验证集
    val_num = args.val_num
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]
    # test
    # input_list_train = input_list_train[:24]
    # input_list_val = input_list_val[:24]

    train_dataset = MyDataset(input_list_train, args.max_len)
    val_dataset = MyDataset(input_list_val, args.max_len)

    return train_dataset, val_dataset

#
# def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
#                 epoch, args):
#     model.train()
#     device = args.device
#     # pad_id = args.pad_id
#     # sep_id = args.sep_id
#     ignore_index = args.ignore_index
#     epoch_start_time = datetime.now()
#     total_loss = 0  # 记录下整个epoch的loss的总和
#
#     # epoch_correct_num:每个epoch中,output预测正确的word的数量
#     # epoch_total_num: 每个epoch中,output预测的word的总数量
#     epoch_correct_num, epoch_total_num = 0, 0
#
#     for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
#         # 捕获cuda out of memory exception
#         try:
#             input_ids = input_ids.to(device)
#             labels = labels.to(device)
#             outputs = model.forward(input_ids, labels=labels)
#             logits = outputs.logits
#             loss = outputs.loss
#             loss = loss.mean()
#
#             # 统计该batch的预测token的正确数与总数
#             batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
#             # 统计该epoch的预测token的正确数与总数
#             epoch_correct_num += batch_correct_num
#             epoch_total_num += batch_total_num
#             # 计算该batch的accuracy
#             batch_acc = batch_correct_num / batch_total_num
#
#             total_loss += loss.item()
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
#
#             loss.backward()
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#
#             # 进行一定step的梯度累计之后，更新参数
#             if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
#                 # 更新参数
#                 optimizer.step()
#                 # 更新学习率
#                 scheduler.step()
#                 # 清空梯度信息
#                 optimizer.zero_grad()
#
#             if (batch_idx + 1) % args.log_step == 0:
#                 logger.info(
#                     "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
#                         batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc, scheduler.get_last_lr()[0]))
#
#             del input_ids, outputs
#
#         except RuntimeError as exception:
#             if "out of memory" in str(exception):
#                 logger.info("WARNING: ran out of memory")
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#             else:
#                 logger.info(str(exception))
#                 raise exception
#
#     # 记录当前epoch的平均loss与accuracy
#     epoch_mean_loss = total_loss / len(train_dataloader)
#     epoch_mean_acc = epoch_correct_num / epoch_total_num
#     logger.info(
#         "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))
#
#     # 保存训练好的模型
#     logger.info('saving model for epoch {}'.format(epoch + 1))
#     model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
#     if not os.path.exists(model_path):
#         os.mkdir(model_path)
#     model_to_save = model.module if hasattr(model, 'module') else model
#     model_to_save.save_pretrained(model_path)
#     logger.info('epoch {} finished'.format(epoch + 1))
#     epoch_finish_time = datetime.now()
#     logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
#
#     return epoch_mean_loss


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args):
    model.train()
    device = args.device
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == 'float16'))  # 根据precision参数初始化GradScaler
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0

    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=(args.precision == 'float16')):  # 根据precision参数使用自动混合精度
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()  # 使用scaler.scale()包裹backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)  # 使用scaler.step()代替optimizer.step()
                scaler.update()  # 更新scaler
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc, scheduler.get_last_lr()[0]))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception
import os
import shutil

def load_checkpoint(model, optimizer, save_model_path):
    start_epoch = 0
    best_val_loss = None
    if os.path.isdir(save_model_path):
        checkpoint_path = os.path.join(save_model_path, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from epoch {start_epoch} with best val loss {best_val_loss}")
    return start_epoch, best_val_loss

def train(model, logger, train_dataset, validate_dataset, args):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info('starting training')

    # Load checkpoint if it exists
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.save_model_path)

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []

    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        if (epoch + 1) % args.save_interval == 0:
            logger.info('每个训练周期（epoch）结束时保存模型的状态 {}'.format(epoch + 1))
            model_path = args.save_model_path  # 直接使用指定的路径，而不是为每个epoch创建新的文件夹
            if os.path.exists(model_path):  # 如果路径已存在，删除旧的检查点
                shutil.rmtree(model_path)
            os.makedirs(model_path, exist_ok=True)  # 使用os.makedirs代替os.mkdir
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            torch.save({
                'model_state_dict': model_to_save.state_dict(),  # 只保存模型参数
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': epoch,
                'best_val_loss': best_val_loss,
            }, join(model_path, 'checkpoint.pth'))
            logger.info('epoch {} finished'.format(epoch + 1))

        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))

def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

def main():
    # 初始化参数
    args = set_args()

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # args.cuda = not args.cuda

    if args.batch_size < 2048 and args.warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    # 创建日志对象
    logger = create_logger(args)

    # # 当用户使用GPU,并且GPU可用时
    # args.cuda = torch.cuda.is_available() and not args.no_cuda
    # device = 'cuda:0' if args.cuda else 'cpu'
    # args.device = device
    # logger.info('using device:{}'.format(device))

    # 上面的代码进行注释，进行了修改，
    # 检查是否设置了使用CUDA，但设备不支持
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("该设备不支持使用CUDA，请尝试使用CPU进行运行")

    # 当用户使用GPU,并且GPU可用时
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))



    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        # model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)

        # 加载自己写的模型model.py
        model = YourModel.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)


    # 根据精度设置模型
    # 由于PyTorch 默认使用的训练精度是 'float32'，
    # 为了加快训练速度，增加数据类型
    if args.precision == 'float16':
        model = model.half()
    elif args.precision == 'bfloat16':
        model = model.bfloat16()

    model = model.to(device)
    # 初始化GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision in ['float16', 'bfloat16']))


    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
        # model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset, validate_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, validate_dataset, args)


if __name__ == '__main__':
    main()
