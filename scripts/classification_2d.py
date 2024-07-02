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
                history=None
                 ):
        model = Classification_2d()
        super().__init__(model, 
                        optimizer_class = optimizer_class, 
                        optimizer_params=optimizer_params,
                        scheduler_class = scheduler_class,
                        scheduler_params=scheduler_params)
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, device, verbose = False):
        inputs, targets = batch
        inputs = inputs[0]
        targets = targets[1]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if verbose:
            print(f"Loss: {loss.item()}")
            print(f"Outputs shape: {outputs.shape}")

        return loss



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
