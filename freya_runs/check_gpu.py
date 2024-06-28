import torch
import time
import os 

if 'freya' in os.uname().nodename:
    print('working of Freya:', os.uname().nodename)

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"CUDA is available. Number of GPUs: {num_gpus}")
        for idx, name in enumerate(gpu_names):
            print(f"GPU {idx}: {name}")
    else:
        print("CUDA is not available.")

check_cuda()


time.sleep(5)
print('finished')
