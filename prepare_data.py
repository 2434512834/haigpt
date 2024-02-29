import os
import numpy as np
import sentencepiece as spm
import torch
import argparse
from sklearn.model_selection import train_test_split

def get_argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess and cache data for GPT training.")
    parser.add_argument("--data_dir", type=str, default="data_dir", help="Directory containing training text files.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path to the tokenizer model file.")
    parser.add_argument("--train_cache_path", type=str, default="data_dir/train_data.pt", help="Path to store processed training data.")
    parser.add_argument("--val_cache_path", type=str, default="data_dir/val_data.pt", help="Path to store processed validation data.")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="Micro batch size per GPU.")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum number of tokens per example.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of data to be used as validation set.")
    return parser

def preprocess_and_cache_data(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    EOS_TOKEN_ID = 2
    all_data = []

    global_input_paths = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, x))]

    for file_path in global_input_paths:
        data = []
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                cc = sp.EncodeAsIds(line.strip()) + [EOS_TOKEN_ID]
                if len(cc) >= 20:
                    data.extend(cc)
                    if len(data) >= args.train_micro_batch_size_per_gpu * (args.max_length + 1):
                        index = args.train_micro_batch_size_per_gpu * (args.max_length + 1)
                        all_data.append(data[:index])
                        data = data[index:]

    if data:
        all_data.append(data)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=args.val_size, random_state=42)

    # Convert to tensor and reshape
    train_tensor_data = [torch.LongTensor(np.asarray(seq).reshape(-1, args.max_length + 1)) for seq in train_data if len(seq) >= args.max_length + 1]
    val_tensor_data = [torch.LongTensor(np.asarray(seq).reshape(-1, args.max_length + 1)) for seq in val_data if len(seq) >= args.max_length + 1]

    # Save processed data
    torch.save(train_tensor_data, args.train_cache_path)
    torch.save(val_tensor_data, args.val_cache_path)

def load_preprocessed_data(train_cache_path, val_cache_path):
    train_data = torch.load(train_cache_path)
    val_data = torch.load(val_cache_path)
    return train_data, val_data

if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    preprocess_and_cache_data(args)
