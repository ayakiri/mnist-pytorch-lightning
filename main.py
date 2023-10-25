import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from simple_net import SimpleNet, SimpleTrainingWithValidation

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())

train_set, val_set, test_set, other = torch.utils.data.random_split(dataset, [5000, 1000, 1000, 53000])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# model
simple_training_with_val = SimpleTrainingWithValidation(SimpleNet())

# Configure tensorboard to monitor training
tensorboard = TensorBoardLogger("tb_logs", name="simple_training_with_val")

# train + validate
trainer = pl.Trainer(logger=tensorboard, max_epochs=100)
trainer.fit(model=simple_training_with_val, train_dataloaders=train_loader, val_dataloaders=val_loader)

# model is overfitted according to tensorboard, but it does work =D
