import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ', device)
if device=='cuda':
    print(torch.cuda.get_device_properties(0).name)

time.sleep(5)
print('finished')
