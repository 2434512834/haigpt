import os
import torch
from models.configuration_baichuan import BaiChuanConfig
from models.modeling_baichuan import BaiChuanForCausalLM

def convert_checkpoint_to_bin(checkpoint_path, config_path, output_dir):
    # 加载配置文件
    config = BaiChuanConfig.from_json_file(config_path)

    # 初始化模型
    model = BaiChuanForCausalLM(config)

    # 加载训练好的检查点
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 保存模型参数为.bin文件
    model.save_pretrained(output_dir)

    print(f"模型已转换并保存到 {output_dir}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch9.pt"  # 训练完成的检查点文件路径
    config_path = "checkpoints/config.json"  # 模型配置文件路径
    output_dir = "converted_model"  # 输出目录

    convert_checkpoint_to_bin(checkpoint_path, config_path, output_dir)
