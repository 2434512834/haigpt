
import torch
import argparse
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import GPT, GPTConfig
from utils import get_params, sample, set_seed, top_k_logits
from dataset import TextDataset, text_collate_fn
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="配置文件的路径")
parser.add_argument("pretrain_model_path", type=str, help="预训练模型的路径")
parser.add_argument("train_data_path", type=str, help="训练数据的路径")
parser.add_argument("val_data_path", type=str, help="验证数据的路径")
parser.add_argument('--use_scheduler', action='store_true', help='是否使用学习率调度器')
parser.add_argument('--early_stopping', action='store_true', help='是否使用早停法')
parser.add_argument('--device', type=str, default='cpu', help='使用的设备，可以是"cpu"或"cuda"')
parser.add_argument('--save_best', action='store_true', help='是否只保存最好的模型')
parser.add_argument('--save_interval', type=int, default=1, help='保存模型的间隔（单位：epoch）')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积的步数')
parser.add_argument('--use_regularization', action='store_true', help='是否使用正则化')
parser.add_argument('--regularization_type', type=str, default='L2', help='正则化类型，可以是"L1"或"L2"')
parser.add_argument('--early_stopping_tolerance', type=int, default=5, help='早停法的容忍度')
parser.add_argument('--gradient_clipping', action='store_true', help='是否使用梯度裁剪')
parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
parser.add_argument('--model_save_path', type=str, default='../../model_fun/model_{epoch}.pt', help='模型保存路径')
parser.add_argument('--regularization_weight', type=float, default=0.01, help='正则化项的权重')
parser.add_argument('--patience', type=int, default=5, help='早停法的耐心值')
args = parser.parse_args()


def train(config_path, pretrain_model_path, train_data_path, val_data_path):
    # 加载配置文件
    config = get_params(config_path)
    set_seed(config.seed)

    # 创建模型
    mconf = GPTConfig(config.vocab_size, config.block_size,
                      n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd)
    model = GPT(mconf)

    # 加载预训练模型
    model.load_state_dict(torch.load(pretrain_model_path))

    # 创建数据加载器
    train_dataset = TextDataset(train_data_path, config.block_size)
    val_dataset = TextDataset(val_data_path, config.block_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=text_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=text_collate_fn)

    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = CrossEntropyLoss()

    # 定义学习率调度器
    if args.use_scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)

    # 定义梯度缩放器（用于混合精度训练）
    if args.mixed_precision:
        scaler = GradScaler()

    # 开始训练
    best_loss = float('inf')
    no_improve_count = 0
    for epoch in range(config.epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            # 前向传播
            with autocast(enabled=args.mixed_precision):
                output = model(batch['input_ids'])
                loss = loss_fn(output.view(-1, config.vocab_size), batch['labels'].view(-1))

                # 如果使用正则化，添加正则化项
                if args.use_regularization:
                    if args.regularization_type == 'L1':
                        l1_regularization = torch.tensor(0.).to(args.device)
                        for param in model.parameters():
                            l1_regularization += torch.norm(param, 1)
                        loss += args.regularization_weight * l1_regularization
                    elif args.regularization_type == 'L2':
                        l2_regularization = torch.tensor(0.).to(args.device)
                        for param in model.parameters():
                            l2_regularization += torch.norm(param, 2)
                        loss += args.regularization_weight * l2_regularization

            # 反向传播
            model.zero_grad()
            if args.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 如果使用梯度裁剪，裁剪梯度
            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 如果使用梯度累积，只有在累积了一定数量的步数后才更新模型参数
            if (i + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if args.use_scheduler:
                    scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['input_ids'])
                loss = loss_fn(output.view(-1, config.vocab_size), batch['labels'].view(-1))

        # 如果使用早停法，当模型的性能在一段时间内没有提升时，提前停止训练
        if args.early_stopping:
            if loss.item() > best_loss:
                no_improve_count += 1
                if no_improve_count >= args.early_stopping_tolerance:
                    break
            else:
                no_improve_count = 0

        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            if args.save_best and loss.item() < best_loss:
                torch.save(model.state_dict(), args.model_save_path.format(epoch=epoch))
                best_loss = loss.item()
            elif not args.save_best:
                torch.save(model.state_dict(), args.model_save_path.format(epoch=epoch))

if __name__ == "__main__":
    train(args.config_path, args.pretrain_model_path, args.train_data_path, args.val_data_path)