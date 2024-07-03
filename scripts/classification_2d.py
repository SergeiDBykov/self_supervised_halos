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

class Classification_2d(nn.Module):
    def __init__(self):
        super(Classification_2d, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class ClassificationModel(BaseModel):
    def __init__(self, 
                optimizer_class=torch.optim.Adam,
                optimizer_params={}, 
                scheduler_class=torch.optim.lr_scheduler.StepLR,
                scheduler_params={},
                criterion=None, 
                history=None,
                transform = None,
                 ):
        model = Classification_2d()
        super().__init__(model, 
                        optimizer_class = optimizer_class, 
                        optimizer_params=optimizer_params,
                        scheduler_class = scheduler_class,
                        scheduler_params=scheduler_params)
        self.criterion = criterion
        self.history = history
        self.transform = transform

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, device, verbose = False):
        inputs, targets = batch
        inputs = inputs[0]
        targets = targets[1]
        inputs, targets = inputs.to(device), targets.to(device)
        if self.transform:
            inputs = self.transform(inputs)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if verbose:
            print(f"Loss: {loss.item()}")
            print(f"Outputs shape: {outputs.shape}")

        return loss

    def show_transforms(self, dataloader, device):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Trial Forward Pass"):
                inputs, targets = batch
                inputs = inputs[0].to(device)
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




def report_classification_performance(model, dataloader, device='cpu', viz_one = False):
    model.model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_masses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Classification Performance"):
            inputs, targets = batch
            image = inputs[0].to(device)
            halo_id = targets[2].to(device)
            halo_mass = targets[0].to(device)
            batch_label = targets[1].to(device)

            pred_class = model(image)


            if viz_one:
                n_img_to_plots = 3
                fig, ax = plt.subplots(1, n_img_to_plots, figsize=(10, 5))
                for i in range(n_img_to_plots):
                    img_input = image[i].cpu().numpy()
                    img_input = np.squeeze(img_input)
                    logmass = halo_mass[i].cpu().numpy()
                    ax[i].imshow(img_input, cmap='afmhot')
                    ax[i].set_title(f"True: {batch_label[i]}, Pred: {torch.argmax(pred_class[i])}; Mass: {logmass:.1f}", fontsize=10)
                return None



            all_preds.append(pred_class)
            all_labels.append(batch_label)
            all_masses.append(halo_mass)
            all_ids.append(halo_id)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_masses = torch.cat(all_masses)
    all_ids = torch.cat(all_ids)


    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    all_ids = all_ids.cpu().numpy()
    all_masses = all_masses.cpu().numpy()


    pred_labels = np.argmax(all_preds, axis=1)

    results_df = pd.DataFrame({'id':all_ids, 'mass':all_masses, 'true_class': all_labels, 'pred_class': pred_labels})

    return results_df
