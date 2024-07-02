from self_supervised_halos.scripts.base_model import BaseModel
import torch.nn as nn




class Classification_2d(BaseModel):
    def __init__(self, optimizer_class, optimizer_params, scheduler_class=None, scheduler_params=None, history = None, device = 'cpu', name = 'classification_2d', augmentations = None):

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

            nn.Flatten(),
            nn.Linear(64, 64),
            nn.Linear(64, 10),
        )

        super(Classification_2d, self).__init__(name, self.cnn, optimizer_class, optimizer_params, scheduler_class, scheduler_params, history)

        self.augmentations = augmentations


    def forward(self, x):
        return self.cnn(x)
    
    def training_step(self, batch, criterion):
        inputs, targets = batch
        inputs_2d = inputs[0]

        if self.augmentations:
            inputs_2d = self.augmentations(inputs_2d)

        outputs = self.model(inputs_2d)
        loss = criterion(outputs, targets)
        return loss

