import pandas as pd
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import time
import os

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


from self_supervised_halos.scripts.base_model import BaseModel


import torch
import numpy as np

def mask_time_series_batch(batch, mask_size=20, num_masks=2, num_masks_var=1, mask_only_nans=False):
    """
    Masks random subsequences of time series data in the batch or masks only NaNs.
    
    Parameters:
    batch (torch.Tensor): Batch of time series data of shape (batch_size, 100).
    mask_size (int): Size of the subsequences to mask.
    num_masks (int): Number of subsequences to mask.
    num_masks_var (int): Variance in the number of subsequences to mask.
    mask_only_nans (bool): If True, only mask NaNs in the input data and produce .
    
    Returns:
    unmasked_signal (torch.Tensor): Original time series data.
    masked_signal (torch.Tensor): Time series data with masked values.
    prediction_mask (torch.Tensor): Mask indicating which tokens need to be predicted.
    """
    batch_size, seq_len = batch.size()
    
    unmasked_signal = batch.clone()
    masked_signal = batch.clone()
    prediction_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    if mask_only_nans:
        # Mask only NaNs in the input data
        masked_signal[torch.isnan(batch)] = float('nan')
        prediction_mask[torch.isnan(batch)] = True
    else:
        for i in range(batch_size):
            available_indices = torch.where(~torch.isnan(batch[i]))[0]
            num_masks_actual = num_masks + np.random.randint(-num_masks_var, num_masks_var + 1)
            
            for _ in range(num_masks_actual):
                if len(available_indices) < mask_size * num_masks_actual:
                    #print('Warning: Not enough available indices to mask. Skipping.')
                    break
                start_idx = np.random.choice(available_indices[:-mask_size + 1])
                mask_indices = torch.arange(start_idx, start_idx + mask_size)
                
                mask_indices = mask_indices[mask_indices < seq_len]
                mask_indices = mask_indices[torch.isin(mask_indices, available_indices)]
                
                masked_signal[i, mask_indices] = float('nan')
                prediction_mask[i, mask_indices] = True

    return unmasked_signal, masked_signal, prediction_mask
    


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=110):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        x = x + self.pe[:x.size(0), :]
        return x


class HaloMassHistTransformer(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, num_layers=3, output_dim=1, dim_feedforward = 32, dropout=0.1):
        super(HaloMassHistTransformer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1)

        self.positional_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout = dropout,
                                                   batch_first = False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(1)  # add channel dimension: (batch_size, 1, seq_len)
        x = self.conv1d(x)  # apply Conv1d: (batch_size, embed_dim, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)

        x = self.positional_encoding(x) # (seq_len, batch_size, embed_dim)

        hidden_states = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # (seq_len, batch_size, embed_dim) 

        x = hidden_states.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        x = self.fc_out(x)  # (batch_size, seq_len, output_dim)
        x = x.squeeze(-1) # remove last dimension
        return x, hidden_states




class RegressionModel(BaseModel):
    def __init__(self, 
                optimizer_class=torch.optim.Adam,
                optimizer_params={}, 
                scheduler_class=torch.optim.lr_scheduler.StepLR,
                scheduler_params={},
                criterion=None, 
                history=None,
                transform = mask_time_series_batch,
                 ):
        model = HaloMassHistTransformer()
        super().__init__(model, 
                        optimizer_class = optimizer_class,
                        optimizer_params=optimizer_params,
                        scheduler_class = scheduler_class,
                        scheduler_params=scheduler_params)
        self.criterion = criterion
        self.history = history if history else {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        self.transform = transform

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)[0]
    
    def training_step(self, batch, device, verbose = False):
        inputs, targets = batch
        time_series = inputs[2].to(device)
        time_series = time_series.float()


        # Mask some subsequences in the batch
        unmasked_signal, masked_signal, prediction_mask = self.transform(time_series)

        # Replace NaNs with zeros for processing
        masked_signal_filled = torch.nan_to_num(masked_signal, nan=-10.0)

        # Create a src_key_padding_mask for the transformer
        #this is for attention mechanism, attention should not be given to the the signal withing the excluded mask and to nan values
        #where src_key_padding_mask is True, the values are ignored in the attention mechanism
        src_key_padding_mask = torch.isnan(masked_signal)

        # Forward pass
        predictions = self.forward(masked_signal_filled, src_key_padding_mask=src_key_padding_mask)
        #predictions = predictions.squeeze(-1) # remove last dimension

        loss = self.criterion(predictions[prediction_mask], unmasked_signal[prediction_mask])

        if verbose:
            print(f"Loss: {loss.item()}")

        return loss


    def show_transforms(self, dataloader, device, plot_n=2):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Trial Forward Pass"):
                inputs, _ = batch
                time_series = inputs[2].to(device)
                time_series = time_series.float()
                if self.transform:
                    unmasked_signal, masked_signal, prediction_mask = self.transform(time_series)
                else:
                    raise NotImplementedError("Transform not implemented")
                        
                masked_signal_filled = torch.nan_to_num(masked_signal, nan=-10.0)

                src_key_padding_mask = torch.isnan(masked_signal)

                predictions = self.forward(masked_signal_filled, src_key_padding_mask=src_key_padding_mask)
                #predictions = predictions.squeeze(-1)
                
                
                fig, ax = plt.subplots(1, plot_n, figsize=(15, 5))
                for i in range(plot_n):
                    #plot masked input
                    input_masked = masked_signal[i].cpu().numpy()
                    pred = predictions[i].cpu().numpy()
                    input_all = time_series[i].cpu().numpy()

                    ax[i].plot(input_masked,  'r--', lw = 6, alpha = 0.3, label='Masked Input')
                    ax[i].plot(pred,  'g-', lw = 3, alpha = 0.5, label='Predictions')
                    ax[i].plot(input_all, 'k:', lw = 3, alpha = 0.5,label='Truth')
                    ax[i].legend()
                plt.show()
                return None

