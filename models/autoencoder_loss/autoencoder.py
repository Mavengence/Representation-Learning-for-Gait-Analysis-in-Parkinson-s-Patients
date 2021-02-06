import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics import Precision, Recall, Accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from pathlib import Path
import numpy as np
import pandas as pd
import os

from argparse import ArgumentParser


class ParkinsonResNetAutoEncoder(LightningModule):
    def __init__(self, learning_rate, max_batch_len, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_batch_len = max_batch_len
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

        self.num_layers = [3, 4, 6, 3]
        self.downblock = Bottleneck
        self.upblock = DeconvBottleneck
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(self.downblock, 64, self.num_layers[0])
        self.layer2 = self._make_downlayer(self.downblock, 128, self.num_layers[1], stride=2)
        self.layer3 = self._make_downlayer(self.downblock, 256, self.num_layers[2], stride=2)
        self.layer4 = self._make_downlayer(self.downblock, 512, self.num_layers[3], stride=2)

        self.uplayer1 = self._make_up_block(self.upblock, 512,  self.num_layers[3], stride=2)
        self.uplayer2 = self._make_up_block(self.upblock, 256, self.num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(self.upblock, 128, self.num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(self.upblock, 64,  self.num_layers[0], stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64)
        )

        self.uplayer_top = DeconvBottleneck(
            self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1, bias=False)

        self.val_losses = []
        self.train_losses = []


    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


    def decoder(self, x, image_size):
        x = self.uplayer4(x)
        x = self.uplayer_top(x)
        x = self.conv1_1(x, output_size=image_size)
        return x


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z, x.size())
        return x_hat


    def training_step(self, batch, batch_idx):
        X, _ = batch
        X = X.view(-1, 1, self.max_batch_len, 12)
        X_hat = self(X)
        
        loss = self.criterion(X_hat, X)
        
        return {"loss": loss}

    
    def validation_step(self, batch, batch_idx):
        X, _ = batch
        X = X.view(-1, 1, self.max_batch_len, 12)
        X_hat = self(X)
        
        loss = self.criterion(X_hat, X)
        
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        X, _ = batch
        X = X.view(-1, 1, self.max_batch_len, 12)
        X_hat = self(X)
        
        loss = self.criterion(X_hat, X)
        
        return {"test_loss": loss}

    
    def training_epoch_end(self, train_step_outputs):
        losses = []
        
        for i in range(len(train_step_outputs)):
            losses.append(float(np.array(train_step_outputs[i]["loss"])))
            
        train_loss = np.mean(losses)
        self.train_losses.append(train_loss)

        self.log('loss', train_loss)


    def validation_epoch_end(self, validation_step_outputs):
        losses = []
        
        for i in range(len(validation_step_outputs)):
            losses.append(float(np.array(validation_step_outputs[i]["val_loss"])))
            
        val_loss = np.mean(losses)
        self.val_losses.append(val_loss)

        self.log('val_loss', val_loss)
  
    
    def test_epoch_end(self, test_step_outputs):
        losses = []
        
        for i in range(len(test_step_outputs)):
            losses.append(float(np.array(test_step_outputs[i]["test_loss"])))
            
        val_loss = np.mean(losses)
        self.val_losses.append(val_loss)

        self.log('test_loss', val_loss)
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(
            block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_batch_len', type=int, default=4000)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=5)
        return parser


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out
    