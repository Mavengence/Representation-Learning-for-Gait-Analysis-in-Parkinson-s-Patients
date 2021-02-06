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

class ParkinsonLSTM(LightningModule):
    def __init__(self, seq_len, max_batch_len, learning_rate, **kwargs):
        super().__init__()
        self.n_features = 12
        self.hidden_size = 256
        self.num_layers = 5
        self.dropout = 0.5
        self.seq_len = seq_len
        self.max_batch_size = max_batch_len
        self.criterion = torch.nn.BCELoss()
        self.learning_rate = learning_rate
        self.batch_size = int(self.max_batch_size / self.seq_len)
        
        self.output_size = 1
        self.fc_size_1 = 50
        self.fc_size_2 = 10

        self.lstm = nn.LSTM(input_size = self.n_features, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers, 
                            dropout = self.dropout, 
                            batch_first = True)
        
        self.activation = nn.LeakyReLU() 
        self.fc1 = nn.Linear(self.hidden_size, self.fc_size_1)
        self.fc2 = nn.Linear(self.fc_size_1, self.fc_size_2)
        self.fc3 = nn.Linear(self.fc_size_2, self.output_size)

        self.sigmoid = nn.Sigmoid()
        
        self.save_hyperparameters()
        self.accuracy = Accuracy()
        #self.roc = pl.metrics.classification.ROC(pos_label=1)
        #self.precision_metric = Precision(num_classes=2)
        #self.recall_metric = Recall(num_classes=2)
        #self.prc = pl.metrics.classification.precision_recall_curve(pos_label=1)
        
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        pack_LSTM_out, hidden = self.lstm(x)
        #print(f"LSTM out: {len(pack_LSTM_out)} - Hidden: {len(hidden)}")
        #y_pred = self.linear(lstm_out[:, -1])
        
        out = hidden[0][-1, ...]  # Get the output of last layer (batch_size, hidden_dim)

        out = self.activation(self.fc1(out))
        #out = self.dropout(out)
        out = self.activation(self.fc2(out))
        #out = F.dropout(out, training=self.training)
        out = self.fc3(out)

        out = self.sigmoid(out)
        #out = out.view(self.seq_len)
        
        return out
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


    def training_step(self, batch, batch_idx):
        X, y = batch[0][0], batch[1][0]
        
        y_hat = self(X)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return {"loss": loss, "train_accuracy": accuracy}

    
    def validation_step(self, batch, batch_idx):
        X, y = batch[0][0], batch[1][0]
        
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        
        accuracy = self.accuracy(y_hat, y)
        #prec = self.precision_metric(y_hat, y)
        #rec = self.recall_metric(y_hat, y)

        return {"loss": loss, "val_accuracy": accuracy}#, "val_precision": prec, "val_recall": rec}
            
    
    def test_step(self, batch, batch_idx):
        X, y = batch[0][0], batch[1][0]
        
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        
        return loss
    
    
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
        #precisions = []
        #recalls = []
        
        for i in range(len(val_step_outputs)):
            losses.append(float(np.array(val_step_outputs[i]["loss"])))
            accuracies.append(float(np.array(val_step_outputs[i]["val_accuracy"]))) 
            #precisions.append(float(np.array(val_step_outputs[i]["val_precision"]))) 
            #recalls.append(float(np.array(val_step_outputs[i]["val_recall"]))) 
            
        self.log('val_loss', np.mean(losses))
        self.log('val_accuracy', np.mean(accuracies))
        #self.log('val_recall', np.mean(recalls))
        #self.log('val_precision', np.mean(precisions))
    
    
    def test_epoch_end(self, test_step_outputs):
        losses = []
        for loss in test_step_outputs:
            losses.append(float(np.array(loss)))
            
        self.log('test_loss', np.mean(losses))
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--seq_len', type=int, default=256)
        parser.add_argument('--max_batch_len', type=int, default=6400)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=5)
        return parser
    