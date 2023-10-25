from torch import nn
import torch
import torchmetrics
import pytorch_lightning as pl


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(),
                                nn.Linear(64, 128), nn.ReLU(),
                                nn.Linear(128, 10))

    def forward(self, x):
        return self.l1(x)


class SimpleTrainingWithValidation(pl.LightningModule):
    def __init__(self, simplenet):
        super().__init__()
        self.simplenet = simplenet
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1) # Spłaszczamy obraz
        y_hat = self.simplenet(x)
        J = self.loss(y_hat, y)
        self.train_acc(y_hat, y)

        self.log('train_loss', J, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return J

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.simplenet(x)
        val_loss = self.loss(y_hat, y)
        self.val_acc(y_hat, y)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
