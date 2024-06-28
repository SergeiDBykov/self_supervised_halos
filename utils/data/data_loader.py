import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from .tng import subhalos_df


class HaloDataset(torch.utils.data.Dataset):
    pass
