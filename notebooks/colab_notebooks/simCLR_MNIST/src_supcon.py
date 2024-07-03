from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE
#import umap #conda install -c conda-forge umap-learn



contrastive_transform = transforms.Compose([
                                transforms.RandomResizedCrop(size=(28, 28), scale = (0.5, 0.95)),
                                transforms.RandomRotation(degrees=45),
                                transforms.ToTensor(),
                                    ])




class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform = contrastive_transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), transforms.ToTensor()(x)]




class SupConLoss(nn.Module): #TODO check/test this loss
    #src: https://github.com/giakoumoglou/classification/tree/main/notebooks
    #if no labels are provided, it is basically the SimCLRLoss #TODO check this with SimCLRLoss class
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,
                 device = 'cpu'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.device = device

    def forward(self, features, labels=None, mask=None):

        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def supcon_train_step(epoch, 
                model, criterion, optimizer, 
                scheduler,
                dataloader_train, 
                history=None, device = 'cpu'):
    
    model.train()

    running_loss = 0.0
    for data, target in dataloader_train:
        batch_size = data[0].shape[0]
        data = torch.cat([data[0], data[1]], dim=0)
        data = data.to(device)



        features = model(data)

        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() if scheduler is not None else None

        running_loss += loss.item()
    
    running_loss = running_loss/len(dataloader_train)
    if history is not None:
        history['train_loss'].append(running_loss)
    
    return running_loss


def supcon_train(epochs,
                model, criterion, optimizer, 
                scheduler,
                dataloader_train, 
                history=None, device = 'cpu'):
    

    pbar_epoch = tqdm(range(epochs), desc='Epochs')

    if history is None:
        history = {'train_loss': [], 'val_loss': []}

    try:
        for epoch in pbar_epoch:
            loss = supcon_train_step(epoch, 
                                    model, criterion, optimizer, 
                                    scheduler,
                                    dataloader_train, 
                                    history, device)
            pbar_epoch.set_postfix({'loss': loss})
    except KeyboardInterrupt:
        print('Interrupted')

    finally:
        torch.save(model.state_dict(), 'models/supcon.pth')
        pd.DataFrame(history).to_csv('models/supcon_history.csv', index=False)
        print('Model and history saved')        

    return history
