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


class BaseModel(nn.Module):
    def __init__(self, name, model, optimizer_class, optimizer_params, 
                 scheduler_class=None, scheduler_params=None,
                 history=None):
        super(BaseModel, self).__init__()
        self.name = name
        self.model = model
        self.optimizer = optimizer_class(self.parameters.parameters(), **optimizer_params)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params) if scheduler_class else None
        
        if history:
            self.history = history
        else:
            self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")
    
    def training_step(self, batch, batch_idx, criterion):
        raise NotImplementedError("Training step not implemented")
    
    def training_loop(self, train_loader, val_loader, num_epochs, criterion, device):
        #for epoch in range(num_epochs):
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            self.train()  # Set the model to training mode
            train_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")):
                loss = self.training_step(batch, batch_idx, criterion)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")
            

            if val_loader:
                self.eval()  # Set the model to evaluation mode
                val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                        inputs, targets = batch
                        outputs = self.model(inputs.to(device))
                        loss = criterion(outputs, targets.to(device))
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
    

    def save(self, models_path, epoch, loss):
        filename = models_path + self.name + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'loss': loss
        }, filename)
        print(f'Model {self.name} saved at epoch {epoch}')

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        loss = checkpoint['loss']
        print(f'Model {self.name} loaded at epoch {checkpoint["epoch"]}')
        return loss





# class BaseHaloModel(nn.Module):

#     def __init__(self, name, model, optimizer, scheduler, history, device):
#         super(BaseHaloModel, self).__init__()
#         self.name = name
#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.history = history
#         self.device = device



#     def save(self, epoch, loss):
#         filename = models_path + self.name + '.pth'
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'history': self.history,
#             'loss': loss
#         }, filename)
#         print(f'Model {self.name} saved at epoch {epoch}')

#     def load(self, filename):
#         checkpoint = torch.load(filename)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.history = checkpoint['history']
#         loss = checkpoint['loss']
#         print(f'Model {self.name} loaded at epoch {checkpoint["epoch"]}')
#         return loss
