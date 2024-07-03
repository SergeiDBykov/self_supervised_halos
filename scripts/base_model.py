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

from self_supervised_halos.utils.utils import res_path
models_path = res_path + 'models/'


class BaseModel:
    def __init__(self, model, 
                 optimizer_class=torch.optim.Adam, optimizer_params={'lr':1e-3}, 
                 scheduler_class=torch.optim.lr_scheduler.StepLR,
                 scheduler_params={'step_size': 20, 'gamma': 0.5},
                 history=None):
        self.model = model
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params) if scheduler_class else None

        self.history = history if history else {'train_loss': [], 'val_loss': [], 'learning_rate': []}

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")

    def training_step(self, batch, device):
        raise NotImplementedError("Training step not implemented")

    def trial_forward_pass(self, dataloader, device, limit_to_first_batch=True):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Trial Forward Pass"):
                self.training_step(batch, device, verbose=True)
                if limit_to_first_batch: break


    def show_transforms(self, dataloader, device):
        raise NotImplementedError("Show transforms not implemented")



    def __call__(self, x):
        return self.forward(x)

    def training_loop(self, train_loader, val_loader, num_epochs, device):
        criterion = self.criterion

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            self.model.train()  # Set the model to training mode
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
                loss = self.training_step(batch, device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            if val_loader:
                self.model.eval()  # Set the model to evaluation mode
                val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                        loss = self.training_step(batch, device)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                self.history['val_loss'].append(avg_val_loss)
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

    def save(self):
        filename = models_path + self.model.__class__.__name__ + '.pth'
        epoch = len(self.history['train_loss'])
        loss = self.history['train_loss'][-1]
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'loss': loss
        }, filename)
        print(f'Model {self.model.__class__.__name__} saved at epoch {epoch}')

    def load(self, filename):
        filename = models_path + filename
        try:
            checkpoint = torch.load(filename)
        except FileNotFoundError:
            print(f"Model {self.model.__class__.__name__}
            not found at {filename}")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        loss = checkpoint['loss']
        print(f'Model {self.model.__class__.__name__} loaded at epoch {checkpoint["epoch"]}')
        return loss
    
