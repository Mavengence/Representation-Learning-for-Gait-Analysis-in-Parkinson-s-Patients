import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything, LightningModule, Trainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.metrics import Precision, Recall, Accuracy

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# weights gewichten von labels
# gradienten visualisieren
# tensorboard logger
# accuracy
# AUC

class ParkinsonAutoEncoder(LightningModule):
    def __init__(self, max_batch_len, learning_rate, **kwargs):
        super().__init__()
        self.input_size = max_batch_len
        self.feature_size = 12
        self.flatten_size = self.input_size * self.feature_size
        self.hidden_size = 128
        self.d = 10
        self.learning_rate = learning_rate

        self.criterion_class = torch.nn.BCELoss()
        self.criterion_ae = torch.nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_size, self.hidden_size), 
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.d))

        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.hidden_size), 
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.flatten_size))

        self.classifier = nn.Sequential(
            nn.Linear(self.d, 1),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

        self.save_hyperparameters()
        self.accuracy = Accuracy()
        
        
    def forward(self, x): 
        z = self.encoder(x)
        y = self.classifier(z)
        return z, y


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(1, self.flatten_size)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.classifier(z)
        
        loss_ae = self.criterion_ae(x_hat, x)
        loss_class = self.criterion_class(y_hat[0], y)
        accuracy = self.accuracy(y_hat[0], y)
        
        loss = loss_ae + loss_class
        
        return {"loss": loss, "train_accuracy": accuracy}

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(1, self.flatten_size)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.classifier(z)
        
        loss_ae = self.criterion_ae(x_hat, x)
        loss_class = self.criterion_class(y_hat[0], y)
        accuracy = self.accuracy(y_hat[0], y)
        
        loss = loss_ae + loss_class
        
        return {"loss": loss, "val_accuracy": accuracy}
    
    
    def training_epoch_end(self, train_step_outputs):
        losses = []
        accuracies = []
        
        for i in range(len(train_step_outputs)):
            losses.append(float(np.array(train_step_outputs[i]["loss"])))
            accuracies.append(float(np.array(train_step_outputs[i]["train_accuracy"]))) 
            
        self.log('train_loss', np.mean(losses))
        self.log('train_accuracy', np.mean(accuracies))
    
    
    def validation_epoch_end(self, val_step_outputs):
        losses = []
        accuracies = []

        
        for i in range(len(val_step_outputs)):
            losses.append(float(np.array(val_step_outputs[i]["loss"])))
            accuracies.append(float(np.array(val_step_outputs[i]["val_accuracy"]))) 
            
        self.log('val_loss', np.mean(losses))
        self.log('val_accuracy', np.mean(accuracies))
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_batch_len', type=int, default=4000)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=5)
        return parser
    