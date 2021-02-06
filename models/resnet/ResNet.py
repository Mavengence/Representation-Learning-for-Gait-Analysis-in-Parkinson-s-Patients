import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torchvision.models as models
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from argparse import ArgumentParser


class ResNet(pl.LightningModule):
    def __init__(self, max_batch_len=4096):
        super().__init__()
        #pytorch lightning object to get the accuracy
        self.acc = pl.metrics.Accuracy()
        self.max_batch_len = max_batch_len
        #Old ResNet from DL course
        self.elu1 = nn.ReLU()
        self.elu2 = nn.ReLU()
        self.elu3 = nn.ReLU()
        self.elu4 = nn.ReLU()
        self.elu5 = nn.ReLU()
        self.elu6 = nn.ReLU()
        self.elu7 = nn.ReLU()
        self.elu8 = nn.ReLU()
        self.elu9 = nn.ReLU()
        self.elu16 = nn.ReLU()
        self.elu17 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.batchnorm10 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.2)
        self.dropout6 = nn.Dropout(0.2)
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc_all_zero = []
        self.train_acc_all_one = []
        self.val_acc_all_zero = []
        self.val_acc_all_one = []
        self.conv1 = nn.Conv2d(1, 64, 12)
        self.maxpool1 = nn.MaxPool1d(3, 3)
        # Resblock 1
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv1x11 = nn.Conv2d(64, 64, 1, 1)
        # Resblock 2
        self.conv4 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv1x12 = nn.Conv2d(64, 128, 1, 2)
        # Resblock 3
        self.conv6 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv1x13 = nn.Conv2d(128, 256, 1, 2)
        # Resblock 4
        self.conv8 = nn.Conv2d(256, 512, 3, 2, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x14 = nn.Conv2d(256, 512, 1, 2)
        # END
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.elu10 = nn.ELU(inplace=False)
        self.fc2 = nn.Linear(512, 512)
        self.elu11 = nn.ELU(inplace=False)
        self.fc3 = nn.Linear(512, 512)
        self.elu12 = nn.ELU(inplace=False)
        self.fc5 = nn.Linear(512, 512)
        self.elu13 = nn.ELU(inplace=False)
        self.fc6 = nn.Linear(512, 512)
        self.elu14 = nn.ELU(inplace=False)
        self.fc7 = nn.Linear(512, 512)
        self.elu15 = nn.ELU(inplace=False)
        self.fc4 = nn.Linear(512, 1)
        self.sigmoid1 = nn.Sigmoid()

        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.elu1(x)
        x = x.view(x.size()[0], x.size()[1], x.size()[2])
        x = self.maxpool1(x)
        x = x.view(x.size()[0], 64, x.size()[2], 1)
        identity = x
        # Resblock 1
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.elu3(x)
        identity = self.conv1x11(identity)
        x += identity
        identity = x
        # Resblock 2
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu4(x)
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.elu5(x)
        identity = self.conv1x12(identity)
        x += identity
        identity = x
        # Resblock 3
        x = self.conv6(x)
        x = self.batchnorm5(x)
        x = self.elu6(x)
        x = self.conv7(x)
        x = self.batchnorm6(x)
        x = self.elu7(x)
        identity = self.conv1x13(identity)
        x += identity
        identity = x
        # Resblock 4
        x = self.conv8(x)
        x = self.batchnorm7(x)
        x = self.elu8(x)
        x = self.conv9(x)
        x = self.batchnorm8(x)
        x = self.elu9(x)
        identity = self.conv1x14(identity)
        x += identity
        x = self.avgpool1(x)
        x = x.flatten(start_dim=1)
        x = self.dropout1(self.fc1(x))
        x = self.elu10(x)
        x = self.dropout2(self.fc2(x))
        x = self.elu11(x)
        x = self.dropout3(self.fc3(x))
        x = self.elu12(x)
        """x = self.dropout4(self.fc5(x))
        x = self.elu13(x)
        x = self.dropout5(self.fc6(x))
        x = self.elu14(x)
        x = self.dropout6(self.fc7(x))
        x = self.elu15(x)"""

        x = self.fc4(x)
        x = self.sigmoid1(x)
        
        
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 8e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(-1, 1, self.max_batch_len, 12)
        # Forward pass
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels.view(-1, 1))
        
        outputs = outputs.cpu()
        labels = labels.cpu()
        acc = (labels == torch.round(outputs)).sum()
        acc = acc / len(outputs)
        zeros = torch.zeros_like(outputs)
        ones = torch.ones_like(outputs)
        acc_zero = (labels == torch.round(zeros)).sum()
        acc_zero = acc_zero / len(outputs)
        acc_one = (labels == torch.round(ones)).sum()
        acc_one = acc_one / len(outputs)
        return {"loss": loss, "acc": acc, "acc_zero": acc_zero, "acc_one": acc_one}
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        data = data.view(-1, 1, self.max_batch_len, 12)
        # Forward pass
        outputs = self(data)

        loss = F.binary_cross_entropy(outputs, labels.view(-1, 1))
        
        outputs = outputs.cpu()
        labels = labels.cpu()
        acc = (labels == torch.round(outputs)).sum()
        acc = acc / len(outputs)
        zeros = torch.zeros_like(outputs)
        ones = torch.ones_like(outputs)
        acc_zero = (labels == torch.round(zeros)).sum()
        acc_zero = acc_zero / len(outputs)
        acc_one = (labels == ones).sum()
        acc_one = acc_one / len(outputs)
        return {"val_loss": loss, "acc": acc, "acc_zero": acc_zero, "acc_one": acc_one}
    """
    def train_dataloader(self):
        dataset = customTrainData()
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True) 
        return train_loader
    
    def val_dataloader(self):
        dataset = customValData()
        val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32) 
        return val_loader
    """
    
    def training_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_acc_zero = torch.stack([x['acc_zero'] for x in outputs]).mean()
        avg_acc_one = torch.stack([x['acc_one'] for x in outputs]).mean()
        self.train_loss.append(avg_loss)
        self.train_acc.append(avg_acc)
        self.train_acc_all_one.append(avg_acc_one)
        self.train_acc_all_zero.append(avg_acc_zero)
    
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        global best_acc
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_acc_zero = torch.stack([x['acc_zero'] for x in outputs]).mean()
        avg_acc_one = torch.stack([x['acc_one'] for x in outputs]).mean()
        self.val_loss.append(avg_loss)
        self.val_acc.append(avg_acc)
        self.val_acc_all_one.append(avg_acc_one)
        self.val_acc_all_zero.append(avg_acc_zero)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_batch_len', type=int, default=4000)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=5)
        return parser