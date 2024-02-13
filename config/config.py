# 训练GPT模型的设置参数，train.py中已经设置默认参数，
# 如果进行了调整，通过命令调整进行传入
# config.py
save_model_path = 'model' # 模型数据文件夹名称
device = '3'
no_cuda = False
vocab_path = '../../vocab/vocab.txt'
model_config = '../../config/config.json'
train_path = '../../data/train.pkl'
max_len = 150
log_path = '../../log/train.log'
log = True
ignore_index = -100
epochs = 100
batch_size = 4
gpu0_bsz = 10
lr = 2.6e-5
eps = 1.0e-09
log_step = 1
gradient_accumulation_steps = 4
max_grad_norm = 2.0
save_model_path = 'model'
pretrained_model = ''
num_workers = 0
patience = 0
warmup_steps = 4000
val_num = 8000




# eval_interval = 2000
# log_interval = 1
# eval_iters = 200
# eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = True # if True, always save a checkpoint after each eval
# init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# # wandb logging
# wandb_log = False # disabled by default
# wandb_project = 'owt'
# wandb_run_name = 'gpt2' # 'run' + str(time.time())
# # data
# dataset = 'openwebtext'
# gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
# batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 1024
# # model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# bias = False # do we use bias inside LayerNorm and Linear layers?
# # adamw optimizer
# learning_rate = 6e-4 # max learning rate
# max_iters = 600000 # total number of training iterations
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# # learning rate decay settings
# decay_lr = True # whether to decay the learning rate
# warmup_iters = 2000 # how many steps to warm up for
# lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# # DDP settings
# backend = 'nccl' # 'nccl', 'gloo', etc.
# # system
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# compile = True # use PyTorch 2.0 to compile the model to be faster

