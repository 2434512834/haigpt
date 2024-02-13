import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    return device

device = check_cuda()
