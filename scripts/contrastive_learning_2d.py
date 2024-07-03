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
from self_supervised_halos.utils.dataloader import img2d_transform


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


class Encoder(nn.Module):

    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()

        image_channels = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128, 64),
            nn.Linear(64, 64),)

    def forward(self, x):
        return self.encoder(x)



class ProjectionHead(nn.Module):
    "Projection head"
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection_head(x)



class SupConNetwork(nn.Module):
    """encoder + projection head"""
    def __init__(self, encoder, head):
        super(SupConNetwork, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(self.head(x), dim=1)
        return x



class ConstrativeLearningModel(BaseModel):
    def __init__(self, 
                optimizer_class=torch.optim.Adam,
                optimizer_params={}, 
                scheduler_class=torch.optim.lr_scheduler.StepLR,
                scheduler_params={},
                criterion=None, 
                history=None,
                transform = img2d_transform,
                use_labels_for_loss = True,
                 ):
        model = SupConNetwork(Encoder(), ProjectionHead())
        super().__init__(model, 
                        optimizer_class = optimizer_class, 
                        optimizer_params=optimizer_params,
                        scheduler_class = scheduler_class,
                        scheduler_params=scheduler_params)
        self.criterion = criterion
        self.history = history if history else {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        self.transform = transform
        self.use_labels_for_loss = use_labels_for_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, device, verbose = False):
        inputs, targets = batch
        images = inputs[0]
        targets = targets[1]


        image_1 = images[0].to(device)
        image_2 = images[1].to(device)
        
        batch_size = image_1.shape[0]

        view_1 = self.transform(image_1)
        view_2 = self.transform(image_2)

        data = torch.cat([view_1, view_2], dim=0)
        data = data.to(device)

        features = self.model(data)

        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if self.use_labels_for_loss:
            loss = self.criterion(features, labels=targets)
        else:
            loss = self.criterion(features)


        if verbose:
            print(f"Loss: {loss.item()}")
            print(f"features shape: {features.shape}")

        return loss

    def show_transforms(self, dataloader, device):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Trial Forward Pass"):
                inputs, targets = batch
                inputs = inputs[0][0].to(device)
                targets = targets[1].to(device)
                if self.transform:
                    inputs_transformed = self.transform(inputs)
                else:
                    print("No transform provided. Showing original images.")
                    inputs_transformed = inputs

                outputs_transformed = self.model(inputs_transformed)
                outputs = self.model(inputs)

                n_img_to_plots = 3
                fig, ax = plt.subplots(2, n_img_to_plots, figsize=(10, 5))
                for i in range(n_img_to_plots):
                    pred_class = torch.argmax(outputs[i])
                    pred_class_transformed = torch.argmax(outputs_transformed[i])
                    img_input = inputs[i].cpu().numpy()
                    img_input = np.squeeze(img_input)
                    img_transformed = inputs_transformed[i].cpu().numpy()
                    img_transformed = np.squeeze(img_transformed)
                    ax[0, i].imshow(img_input, cmap='afmhot')
                    ax[0, i].set_title(f"INPUT: True: {targets[i]}, Pred: {pred_class}", fontsize=10)
                    ax[1, i].imshow(img_transformed, cmap='afmhot')
                    ax[1, i].set_title(f"TRANSFORMED: True: {targets[i]}, Pred: {pred_class_transformed}", fontsize=10)
                    #remove axis
                    ax[0, i].axis('off')
                    ax[1, i].axis('off')

                plt.show()
                return inputs,inputs_transformed



