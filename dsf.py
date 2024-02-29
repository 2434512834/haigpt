import argparse
import time
from transformers import AutoTokenizer
from models.modeling_baichuan import BaiChuanForCausalLM

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="你好\n你好吗？->", help="输入文本的前缀或提示")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成文本的最大长度")
    parser.add_argument("--min_length", type=int, default=None, help="生成文本的最小长度")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成文本的温度值")
    parser.add_argument("--top_k", type=int, default=0, help="每个时间步考虑的最大标记数量")
    parser.add_argument("--top_p", type=float, default=1.0, help="累计概率阈值")
    parser.add_argument("--num_beams", type=int, default=1, help="beam search的beam数量")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="返回的候选文本数量")
    parser.add_argument("--do_sample", action="store_true", help="是否使用随机采样")
    # 新增参数
    parser.add_argument("--length_penalty", type=float, default=1.0, help="长度惩罚系数")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="禁止重复的n-gram大小")
    parser.add_argument("--early_stopping", action="store_true", help="是否提前停止生成")
    return parser

def generate_text(args):
    # 开始计时
    start_time = time.time()

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("/home/shidonghai/haigpt/train/train_02/converted_model", trust_remote_code=True)
    model = BaiChuanForCausalLM.from_pretrained("/home/shidonghai/haigpt/train/train_02/converted_model")
    model = model.to('cuda:0')


    # 如果命令行参数中没有提供输入文本，则使用代码中指定的默认文本
    prompt = args.prompt if args.prompt is not None else "你好\n你今天吃饭了吗？->"

    # 准备输入文本
    inputs = tokenizer(args.prompt, return_tensors='pt')
    inputs = inputs.to('cuda:0')

    # 设置生成参数
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_length": args.min_length,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": args.early_stopping,
    }
    # 移除值为None的键
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    # 生成预测
    pred = model.generate(**inputs, **generate_kwargs)

    # 解码并打印预测结果
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

    # 结束计时并打印耗时
    end_time = time.time()
    print(f"生成耗时：{end_time - start_time}秒")

if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()
    generate_text(args)
