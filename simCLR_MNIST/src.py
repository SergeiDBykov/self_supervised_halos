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


LOAD_MODELS = True


original_transform = transforms.Compose([
    transforms.ToTensor()
])


contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=(28, 28), scale = (0.5, 0.95)),
    transforms.RandomRotation(degrees=45),
])




def get_mnist_dataloaders(batch_size=128, train_prop=0.7, val_prop=0.2, dataset_size=1.0):
    # Ensure that the proportions sum to 1
    assert train_prop + val_prop <= 1, "Train and validation proportions must sum to less than or equal to 1."
    assert 0 < dataset_size <= 1, "Dataset size must be between 0 and 1."

    # Load the MNIST dataset
    full_dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=original_transform)

    # Shuffle the dataset and select a fraction of it
    full_length = len(full_dataset)
    indices = torch.randperm(full_length).tolist()
    dataset = torch.utils.data.Subset(full_dataset, indices[:int(full_length * dataset_size)])

    # Calculate the lengths of each split
    train_length = int(train_prop * len(dataset))
    val_length = int(val_prop * len(dataset))
    test_length = len(dataset) - train_length - val_length

    # Split the dataset
    dataset_train, dataset_val, dataset_test = random_split(dataset, [train_length, val_length, test_length])

    #print dataset sizes
    print(f'Train size: {len(dataset_train)}')
    print(f'Validation size: {len(dataset_val)}')
    print(f'Test size: {len(dataset_test)}')


    # Create data loaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test


class SimCLRLoss(nn.Module): #TODO check/test this loss
    #https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    def __init__(self, temperature=0.5, device = 'cpu'):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, zis, zjs):
        batch_size = zis.size(0)

        # Normalize the embeddings
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Compute similarity
        representations = torch.cat([zis, zjs], dim=0)
        similarity_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))

        # Create the labels
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Mask to remove positive samples from the similarity matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create target labels
        target = torch.arange(batch_size).to(self.device) #used to be .to(labels.device) and it DID not work! no idea why

        loss_i = self.criterion(similarity_matrix[:batch_size], target)
        loss_j = self.criterion(similarity_matrix[batch_size:], target)

        loss = (loss_i + loss_j) / 2
        return loss



def simCLR_train(model, criterion, optimizer, scheduler,
                dataloader_train, dataloader_val=None,
                history=None,
                epochs = 5, device = 'cpu',
                contrast_transforms = contrast_transforms):
    
    pbar_epoch = tqdm(range(epochs), desc='Epochs')

    if history is None:
        history = {'train_loss': [], 'val_loss': []}


    try:

        for epoch in pbar_epoch:
            running_loss = 0.0

            for images, _ in dataloader_train:
                images = images.to(device)
                view1 = contrast_transforms(images)
                view2 = contrast_transforms(images)

                batch_size_curr = images.size(0)

                images = torch.cat([view1, view2], dim=0).to(device)

                zis = model(images[:batch_size_curr])
                zjs = model(images[batch_size_curr:])

                loss = criterion(zis, zjs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


                running_loss += loss.item() * batch_size_curr

            running_loss /= len(dataloader_train.dataset)
            if dataloader_val is None:
                pbar_epoch.set_postfix({'Loss': f'{running_loss:.5f}'})


            if dataloader_val is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, _ in dataloader_val:
                        images = images.to(device)
                        view1 = contrast_transforms(images)
                        view2 = contrast_transforms(images)

                        batch_size_curr = images.size(0)

                        images = torch.cat([view1, view2], dim=0).to(device)

                        zis = model(images[:batch_size_curr])
                        zjs = model(images[batch_size_curr:])

                        loss = criterion(zis, zjs)

                        val_loss += loss.item() * batch_size_curr

                model.train()
                val_loss /= len(dataloader_val.dataset)
                pbar_epoch.set_postfix({'Loss': f'{running_loss / len(dataloader_train.dataset):.5f}', 'Val Loss': f'{val_loss:.5f}'})


            history['train_loss'].append(running_loss)
            history['val_loss'].append(val_loss if dataloader_val is not None else -1.0)

    except KeyboardInterrupt:
        print('Training interrupted by user')
    
    finally:
        if LOAD_MODELS:
            torch.save(model.state_dict(), 'models/simclr')
            pd.DataFrame(history).to_csv('models/simclr_history.csv', index=False)
            print('Model and history saved')
    

    return history



def downstream_train(classification_model,
                    simclr_model,
                    criterion, optimizer, scheduler,
                    dataloader_train, dataloader_val=None,
                    history=None,
                    epochs = 5, device = 'cpu'):
    

    pbar_epoch = tqdm(range(epochs), desc='Epochs')

    if history is None:
        history = {'train_loss': [], 'val_loss': []}


    try:

        for epoch in pbar_epoch:
            running_loss = 0.0

            for images, labels in dataloader_train:
                images = images.to(device)

                with torch.no_grad():
                    representations = simclr_model.encoder(images)

                outputs = classification_model(representations)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * batch_size_curr

            running_loss /= len(dataloader_train.dataset)
            if dataloader_val is None:
                pbar_epoch.set_postfix({'Loss': f'{running_loss:.5f}'})


            if dataloader_val is not None:
                classification_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in dataloader_val:
                        images = images.to(device)
                        batch_size_curr = images.size(0)

                        with torch.no_grad():
                            representations = simclr_model.encoder(images)
                        
                        outputs = classification_model(representations)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * batch_size_curr


                classification_model.train()
                val_loss /= len(dataloader_val.dataset)
                pbar_epoch.set_postfix({'Loss': f'{running_loss / len(dataloader_train.dataset):.5f}', 'Val Loss': f'{val_loss:.5f}'})


            history['train_loss'].append(running_loss)
            history['val_loss'].append(val_loss if dataloader_val is not None else -1.0)

    except KeyboardInterrupt:
        print('Training interrupted by user')
    
    finally:
        if LOAD_MODELS:
            torch.save(classification_model.state_dict(), 'models/downstream')
            pd.DataFrame(history).to_csv('models/downstream_history.csv', index=False)
            print('Model and history saved')
    

    return history

                     



def visualize_batch(dataloader, n=8,  transform=contrast_transforms):

    batch, _ = next(iter(dataloader))

    fig, axs = plt.subplots(8, 3, figsize=(3, 8))

    for i in range(n):
        original = batch[i]
        axs[i, 0].imshow(original[0].numpy(), cmap='gray')
        if i==0:
            axs[i, 0].set_title('Original')
        axs[i, 0].axis('off')

        for j in range(1, 3):
            augmented = transform(original)
            axs[i, j].imshow(augmented[0].numpy(), cmap='gray')
            axs[i, j].axis('off')
            if i==0:
                axs[i, j].set_title('View {}'.format(j))

    #remove space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
